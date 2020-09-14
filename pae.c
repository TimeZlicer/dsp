#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "pae.h"
#include "util.h"

#define PAD_CHUNK_LEN 4096   /* zero-padded chunk length */
#define SHIFT_STEP       2
#define N_SHIFT         22   /* ±samples*SHIFT_STEP (must be even if OVERLAP==75) */
#define SHIFT_WEIGHT     4   /* must be an even integer >=2 */
#define OVERLAP         75   /* window overlap; 50% and 75% are supported */
#define N_BANDS         32
#define OMEGA_THRESHOLD 0.3

#define DEBUG_PCA 0
#if DEBUG_PCA
	#define PCA_DEBUG(...) fprintf(stderr, __VA_ARGS__)
#else
	#define PCA_DEBUG(...)
#endif

#if OVERLAP == 75
	#define HOP (CHUNK_LEN / 4)
	#define WINDOW_NORM 0.5
#elif OVERLAP == 50
	#define HOP (CHUNK_LEN / 2)
	#define WINDOW_NORM 1.0
#else
	#error "invalid OVERLAP"
#endif

#define CHUNK_LEN (PAD_CHUNK_LEN - N_SHIFT*SHIFT_STEP*2)
#define CHUNK_FR_LEN (PAD_CHUNK_LEN / 2 + 1)
#define BAND_LEN ((CHUNK_FR_LEN-1) / N_BANDS)
#if N_SHIFT == 0
	#define FD_DELAY(x, f, t) (x)
#else
	#define FD_DELAY(x, f, t) ((x)*cexp(-(I*2.0*M_PI*(f)*(t))))
#endif

struct time_shift {
	fftw_complex r00, r11, r01, eig0, eig1;
	double weight;
};

struct pae_state {
	int c0, c1, has_output, is_draining;
	ssize_t in_buf_pos, out_buf_pos, um_buf_pos, drain_pos, drain_frames;
	sample_t *window;
	sample_t *input[2], *output[4], *tmp, **um_buf;
	fftw_complex *in_fr[2], *out_fr[4];
	fftw_plan r2c_plan[2], c2r_plan[4];
};

static void process_chunk_mspca(struct effect *e, struct pae_state *state, ssize_t start)
{
	struct time_shift ts[N_SHIFT*2+1];
	ssize_t k, j;
	/* Calculate correlations, eigenvalues, and weights for each time shift */
	double weight_sum = 0.0;
	for (j = 0; j < N_SHIFT*2+1; ++j) {
		double delay = ((double) j - N_SHIFT) / e->istream.fs * SHIFT_STEP;  /* time delay in seconds */
		ts[j].r00 = 0.0;  /* auto-correlation of channel 0 */
		ts[j].r11 = 0.0;  /* auto-correlation of channel 1 */
		ts[j].r01 = 0.0;  /* cross-correlation of channels 0 and 1 */
		for (k = start; k < start+BAND_LEN; ++k) {
			double f = (double) k / PAD_CHUNK_LEN * e->istream.fs;
			fftw_complex x0 = FD_DELAY(state->in_fr[0][k], f, delay);
			fftw_complex x1 = state->in_fr[1][k];
			ts[j].r00 += conj(x0)*x0;
			ts[j].r11 += conj(x1)*x1;
			ts[j].r01 += conj(x0)*x1;
		}
		PCA_DEBUG("r00    = %g%+gi\n", creal(ts[j].r00), cimag(ts[j].r00));
		PCA_DEBUG("r11    = %g%+gi\n", creal(ts[j].r11), cimag(ts[j].r11));
		PCA_DEBUG("r01    = %g%+gi\n", creal(ts[j].r01), cimag(ts[j].r01));
		/* Find two eigenvalues using the quadratic formula */
		fftw_complex b = -(ts[j].r00)-(ts[j].r11), c = (ts[j].r00)*(ts[j].r11)-(ts[j].r01)*(ts[j].r01);
		ts[j].eig0 = (-b + csqrt(b*b - 4.0*c)) / 2.0;
		ts[j].eig1 = (-b - csqrt(b*b - 4.0*c)) / 2.0;
		PCA_DEBUG("eig0   = %g%+gi\n", creal(ts[j].eig0), cimag(ts[j].eig0));
		PCA_DEBUG("eig1   = %g%+gi\n", creal(ts[j].eig1), cimag(ts[j].eig1));
		ts[j].weight = pow(creal(ts[j].r01), SHIFT_WEIGHT);
		weight_sum += ts[j].weight;
	}
	/* Normalize weights */
	for (j = 0; j < N_SHIFT*2+1; ++j) {
		ts[j].weight /= weight_sum;
		PCA_DEBUG("weight[%zd] = %g\n", j, ts[j].weight);
	}
	/* Zero the output arrays */
	for (k = start; k < start+BAND_LEN; ++k) {
		state->out_fr[0][k] = 0.0;
		state->out_fr[1][k] = 0.0;
		state->out_fr[2][k] = 0.0;
		state->out_fr[3][k] = 0.0;
	}
	for (j = 0; j < N_SHIFT*2+1; ++j) {
		double delay = ((double) j - N_SHIFT) / e->istream.fs * SHIFT_STEP;  /* time delay in seconds */
		double omega = creal(1.0 - cabs((ts[j].eig1)/(ts[j].eig0))); /* ratio of primary to ambient components */
		PCA_DEBUG("omega  = %g\n", omega);
		if (ts[j].r01 == 0.0 && (ts[j].r00 == 0.0 || ts[j].r11 == 0.0)) {
			/* Only the primary component is present and is hard panned */
			for (k = start; k < start+BAND_LEN; ++k) {
				state->out_fr[0][k] += state->in_fr[0][k] * ts[j].weight;
				state->out_fr[1][k] += state->in_fr[1][k] * ts[j].weight;
			}
		}
		else if (omega < OMEGA_THRESHOLD) {
			/* Assume that there is no actual primary component for this band */
			for (k = start; k < start+BAND_LEN; ++k) {
				state->out_fr[2][k] += state->in_fr[0][k] * ts[j].weight;
				state->out_fr[3][k] += state->in_fr[1][k] * ts[j].weight;
			}
		}
		else {
			/* Calculate primary and ambient components */
			fftw_complex pan = ((ts[j].eig0)-(ts[j].r00)) / (ts[j].r01);  /* panning factor */
			for (k = start; k < start+BAND_LEN; ++k) {
				double f = (double) k / PAD_CHUNK_LEN * e->istream.fs;
				fftw_complex x0 = FD_DELAY(state->in_fr[0][k], f, delay);
				fftw_complex x1 = state->in_fr[1][k];
				fftw_complex p0 = (x0 + pan*x1) / (1.0 + pan*pan);
				fftw_complex p1 = pan*p0;
				fftw_complex a0 = pan * (pan*x0 - x1) / (1.0 + pan*pan);
				fftw_complex a1 = -a0/pan;

				state->out_fr[0][k] += FD_DELAY(p0, f, -delay) * ts[j].weight;
				state->out_fr[1][k] += p1 * ts[j].weight;
				state->out_fr[2][k] += FD_DELAY(a0, f, -delay) * ts[j].weight;
				state->out_fr[3][k] += a1 * ts[j].weight;
				PCA_DEBUG("p0=%g%+gi; p1=%g%+gi; a0=%g%+gi; a1=%g%+gi\n",
					creal(p0), cimag(p0), creal(p1), cimag(p1), creal(a0), cimag(a0), creal(a1), cimag(a1));
			}
		}
	}
	PCA_DEBUG("---------------------------------\n\n");
}

sample_t * pae_effect_run(struct effect *e, ssize_t *frames, sample_t *ibuf, sample_t *obuf)
{
	struct pae_state *state = (struct pae_state *) e->data;
	ssize_t i, k, j, iframes = 0, oframes = 0;
	while (iframes < *frames) {
		while (state->in_buf_pos < CHUNK_LEN && iframes < *frames) {
			for (i = 0, k = 0, j = 0; i < e->istream.channels; ++i, ++k) {
				if (i == state->c0) {
					obuf[oframes * e->ostream.channels + k++] = (state->has_output) ? state->output[0][state->out_buf_pos] : 0;
					obuf[oframes * e->ostream.channels + k] = (state->has_output) ? state->output[1][state->out_buf_pos] : 0;
					state->input[0][state->in_buf_pos] = (ibuf) ? ibuf[iframes * e->istream.channels + i] : 0;
				}
				else if (i == state->c1) {
					obuf[oframes * e->ostream.channels + k++] = (state->has_output) ? state->output[2][state->out_buf_pos] : 0;
					obuf[oframes * e->ostream.channels + k] = (state->has_output) ? state->output[3][state->out_buf_pos] : 0;
					state->input[1][state->in_buf_pos] = (ibuf) ? ibuf[iframes * e->istream.channels + i] : 0;
				}
				else {
					obuf[oframes * e->ostream.channels + k] = (state->has_output) ? state->um_buf[j][state->um_buf_pos] : 0;
					state->um_buf[j][state->um_buf_pos] = (ibuf) ? ibuf[iframes * e->istream.channels + i] : 0;
					++j;
				}
			}
			#ifdef SYMMETRIC_IO
				++oframes;
			#else
				if (state->has_output)
					++oframes;
			#endif
			++iframes;
			++state->in_buf_pos;
			++state->out_buf_pos;
			state->um_buf_pos = (state->um_buf_pos + 1 == CHUNK_LEN) ? 0 : state->um_buf_pos + 1;
		}

		if (state->in_buf_pos == CHUNK_LEN) {
			for (i = 0; i < 2; ++i) {
				memset(state->tmp, 0, PAD_CHUNK_LEN * sizeof(sample_t));
				memcpy(&(state->tmp[N_SHIFT*SHIFT_STEP]), state->input[i], CHUNK_LEN * sizeof(sample_t));
				memmove(state->input[i], &(state->input[i][HOP]), (CHUNK_LEN - HOP) * sizeof(sample_t));
				for (k = 0; k < CHUNK_LEN; ++k)
					state->tmp[k+N_SHIFT*SHIFT_STEP] *= state->window[k];
				fftw_execute(state->r2c_plan[i]);
			}

			/* Primary-ambient extraction using modified PCA */
			/* TODO: use a dynamic partitioning scheme */

			/* Pass DC component directly */
			state->out_fr[0][0] = state->in_fr[0][0];
			state->out_fr[1][0] = state->in_fr[1][0];
			state->out_fr[2][0] = 0.0;
			state->out_fr[3][0] = 0.0;

			#if defined(_OPENMP) && !(DEBUG_PCA)
				#pragma omp parallel for schedule(static)
			#endif
			for (i = 0; i < N_BANDS; ++i)
				process_chunk_mspca(e, state, i*BAND_LEN + 1);

			for (i = 0; i < 4; ++i) {
				memmove(state->output[i], &(state->output[i][HOP]), (CHUNK_LEN - HOP) * sizeof(sample_t));
				memset(&(state->output[i][CHUNK_LEN - HOP]), 0, HOP * sizeof(sample_t));
				fftw_execute(state->c2r_plan[i]);
				for (k = 0; k < CHUNK_LEN; ++k)
					state->output[i][k] += state->tmp[k+N_SHIFT*SHIFT_STEP] / PAD_CHUNK_LEN * state->window[k];
			}
			state->in_buf_pos = CHUNK_LEN - HOP;
			state->out_buf_pos = 0;
			state->has_output = 1;
		}
	}
	*frames = oframes;
	return obuf;
}

ssize_t pae_effect_delay(struct effect *e)
{
	struct pae_state *state = (struct pae_state *) e->data;
	return (state->has_output) ? CHUNK_LEN : state->in_buf_pos;
}

void pae_effect_reset(struct effect *e)
{
	int i;
	struct pae_state *state = (struct pae_state *) e->data;
	state->in_buf_pos = 0;
	state->out_buf_pos = 0;
	state->um_buf_pos = 0;
	state->has_output = 0;
	for (i = 0; i < 4; ++i)
		memset(state->output[i], 0, CHUNK_LEN * sizeof(sample_t));
	for (i = 0; i < e->istream.channels - 2; ++i)
		memset(state->um_buf[i], 0, CHUNK_LEN * sizeof(sample_t));
}

void pae_effect_drain(struct effect *e, ssize_t *frames, sample_t *obuf)
{
	struct pae_state *state = (struct pae_state *) e->data;
	if (!state->has_output && state->out_buf_pos == 0)
		*frames = -1;
	else {
		if (!state->is_draining) {
			state->drain_frames = HOP;
			if (state->has_output)
				state->drain_frames += CHUNK_LEN - HOP - state->out_buf_pos;
			state->drain_frames += state->out_buf_pos;
			state->is_draining = 1;
		}
		if (state->drain_pos < state->drain_frames) {
			pae_effect_run(e, frames, NULL, obuf);
			state->drain_pos += *frames;
			*frames -= (state->drain_pos > state->drain_frames) ? state->drain_pos - state->drain_frames : 0;
		}
		else
			*frames = -1;
	}
}

void pae_effect_destroy(struct effect *e)
{
	int i;
	struct pae_state *state = (struct pae_state *) e->data;
	free(state->window);
	fftw_free(state->tmp);
	for (i = 0; i < 2; ++i) {
		fftw_free(state->input[i]);
		fftw_free(state->in_fr[i]);
		fftw_free(state->r2c_plan[i]);
	}
	for (i = 0; i < 4; ++i) {
		fftw_free(state->output[i]);
		fftw_free(state->out_fr[i]);
		fftw_free(state->c2r_plan[i]);
	}
	for (i = 0; i < e->istream.channels - 2; ++i)
		free(state->um_buf[i]);
	free(state->um_buf);
	free(state);
}

struct effect * pae_effect_init(struct effect_info *ei, struct stream_info *istream, char *channel_selector, const char *dir, int argc, char **argv)
{
	int i, n_channels = 0;
	struct effect *e;
	struct pae_state *state;

	if (argc != 1) {
		LOG_FMT(LL_ERROR, "%s: usage: %s", argv[0], ei->usage);
		return NULL;
	}
	if (!(istream->fs == 44100 || istream->fs == 48000)) {
		LOG_FMT(LL_ERROR, "%s: error: sample rate must be 44100 or 48000", argv[0]);
		return NULL;
	}
	for (i = 0; i < istream->channels; ++i)
		if (GET_BIT(channel_selector, i))
			++n_channels;
	if (n_channels != 2) {
		LOG_FMT(LL_ERROR, "%s: error: number of input channels must be 2", argv[0]);
		return NULL;
	}

	e = calloc(1, sizeof(struct effect));
	e->name = ei->name;
	e->istream.fs = e->ostream.fs = istream->fs;
	e->istream.channels = istream->channels;
	e->ostream.channels = istream->channels + 2;
	e->run = pae_effect_run;
	e->delay = pae_effect_delay;
	e->reset = pae_effect_reset;
	e->drain = pae_effect_drain;
	e->destroy = pae_effect_destroy;

	state = calloc(1, sizeof(struct pae_state));
	e->data = state;

	state->c0 = state->c1 = -1;
	for (i = 0; i < istream->channels; ++i) {
		if (GET_BIT(channel_selector, i)) {
			if (state->c0 == -1)
				state->c0 = i;
			else
				state->c1 = i;
		}
	}
	state->window = calloc(CHUNK_LEN, sizeof(sample_t));
	for (i = 0; i < CHUNK_LEN; ++i)
		state->window[i] = sqrt(WINDOW_NORM * 0.5 * (1.0 - cos(2.0*M_PI*i / CHUNK_LEN)));  /* root-hann window */
	state->tmp = fftw_malloc(PAD_CHUNK_LEN * sizeof(sample_t));
	memset(state->tmp, 0, PAD_CHUNK_LEN * sizeof(sample_t));
	for (i = 0; i < 2; ++i) {
		state->input[i] = fftw_malloc(CHUNK_LEN * sizeof(sample_t));
		memset(state->input[i], 0, CHUNK_LEN * sizeof(sample_t));
		state->in_fr[i] = fftw_malloc(CHUNK_FR_LEN * sizeof(fftw_complex));
		memset(state->in_fr[i], 0, CHUNK_FR_LEN * sizeof(fftw_complex));
		state->r2c_plan[i] = fftw_plan_dft_r2c_1d(PAD_CHUNK_LEN, state->tmp, state->in_fr[i], FFTW_ESTIMATE);
	}
	for (i = 0; i < 4; ++i) {
		state->output[i] = fftw_malloc(CHUNK_LEN * sizeof(sample_t));
		memset(state->output[i], 0, CHUNK_LEN * sizeof(sample_t));
		state->out_fr[i] = fftw_malloc(CHUNK_FR_LEN * sizeof(fftw_complex));
		memset(state->out_fr[i], 0, CHUNK_FR_LEN * sizeof(fftw_complex));
		state->c2r_plan[i] = fftw_plan_dft_c2r_1d(PAD_CHUNK_LEN, state->out_fr[i], state->tmp, FFTW_ESTIMATE);
	}
	if (e->istream.channels > 2) {
		state->um_buf = calloc(e->istream.channels - 2, sizeof(sample_t *));
		for (i = 0; i < e->istream.channels - 2; ++i)
			state->um_buf[i] = calloc(CHUNK_LEN, sizeof(sample_t));
	}

	return e;
}
