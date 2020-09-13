#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "pae.h"
#include "util.h"

#define CHUNK_LEN 4096
#define OVERLAP 75
#define CHUNK_FR_LEN (CHUNK_LEN / 2 + 1)
#define BAND_LEN ((CHUNK_FR_LEN-1) / 32)
#define OMEGA_THRESHOLD 0.4

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

struct pae_state {
	int c0, c1, has_output, is_draining;
	ssize_t in_buf_pos, out_buf_pos, um_buf_pos, drain_pos, drain_frames;
	sample_t *window;
	sample_t *input[2], *output[4], *tmp, **um_buf;
	fftw_complex *in_fr[2], *out_fr[4];
	fftw_plan r2c_plan[2], c2r_plan[4];
};

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
				memcpy(state->tmp, state->input[i], CHUNK_LEN * sizeof(sample_t));
				memmove(state->input[i], &(state->input[i][HOP]), (CHUNK_LEN - HOP) * sizeof(sample_t));
				for (k = 0; k < CHUNK_LEN; ++k)
					state->tmp[k] *= state->window[k];
				fftw_execute(state->r2c_plan[i]);
			}

			/* Primary-ambient extraction using modified PCA */
			/* TODO: detect inter-channel time differences */
			/* TODO: use a dynamic partitioning scheme */

			/* Pass DC component directly */
			state->out_fr[0][0] = state->in_fr[0][0];
			state->out_fr[1][0] = state->in_fr[1][0];
			state->out_fr[2][0] = 0.0;
			state->out_fr[3][0] = 0.0;

			for (i = 1; i < CHUNK_FR_LEN; i+=BAND_LEN) {
				fftw_complex r00 = 0.0;  /* auto-correlation of channel 0 */
				fftw_complex r11 = 0.0;  /* auto-correlation of channel 1 */
				fftw_complex r01 = 0.0;  /* cross-correlation of channels 0 and 1 */
				for (k = i; k < i+BAND_LEN; ++k) {
					fftw_complex x0 = state->in_fr[0][k];
					fftw_complex x1 = state->in_fr[1][k];
					r00 += conj(x0)*x0;
					r11 += conj(x1)*x1;
					r01 += conj(x0)*x1;
				}
				PCA_DEBUG("r00    = %g%+gi\n", creal(r00), cimag(r00));
				PCA_DEBUG("r11    = %g%+gi\n", creal(r11), cimag(r11));
				PCA_DEBUG("r01    = %g%+gi\n", creal(r01), cimag(r01));
				/* find two eigenvalues using the quadratic formula */
				fftw_complex b = -r00-r11, c = r00*r11-r01*r01;
				fftw_complex eig[2] = {
					(-b + csqrt(b*b - 4.0*c)) / 2.0,
					(-b - csqrt(b*b - 4.0*c)) / 2.0
				};
				PCA_DEBUG("eig[0] = %g%+gi\n", creal(eig[0]), cimag(eig[0]));
				PCA_DEBUG("eig[1] = %g%+gi\n", creal(eig[1]), cimag(eig[1]));
				/* ratio of primary to ambient components */
				double omega = creal(1.0 - cabs(eig[1]/eig[0]));
				PCA_DEBUG("omega  = %g\n", omega);
				if (omega < OMEGA_THRESHOLD) {
					/* assume that there is no actual primary component for this band */
					for (k = i; k < i+BAND_LEN; ++k) {
						state->out_fr[0][k] = 0.0;
						state->out_fr[1][k] = 0.0;
						state->out_fr[2][k] = state->in_fr[0][k];
						state->out_fr[3][k] = state->in_fr[1][k];
					}
				}
				else {
					/* calculate primary and ambient components */
					for (k = i; k < i+BAND_LEN; ++k) {
						fftw_complex x0 = state->in_fr[0][k];
						fftw_complex x1 = state->in_fr[1][k];
						fftw_complex pan = (r01 == 0.0) ? 1.0 : (eig[0]-r00) / r01;  /* panning factor */
						fftw_complex p0 = (x0 + pan*x1) / (1.0 + pan*pan);
						fftw_complex p1 = pan*p0;
						fftw_complex a0 = pan * (pan*x0 - x1) / (1.0 + pan*pan);
						fftw_complex a1 = -a0/pan;

						state->out_fr[0][k] = p0;
						state->out_fr[1][k] = p1;
						state->out_fr[2][k] = a0;
						state->out_fr[3][k] = a1;
						PCA_DEBUG("p0=%g%+gi; p1=%g%+gi; a0=%g%+gi; a1=%g%+gi\n",
							creal(p0), cimag(p0), creal(p1), cimag(p1), creal(a0), cimag(a0), creal(a1), cimag(a1));
					}
				}
				PCA_DEBUG("---------------------------------\n\n");
			}
			for (i = 0; i < 4; ++i) {
				memmove(state->output[i], &(state->output[i][HOP]), (CHUNK_LEN - HOP) * sizeof(sample_t));
				memset(&(state->output[i][CHUNK_LEN - HOP]), 0, HOP * sizeof(sample_t));
				fftw_execute(state->c2r_plan[i]);
				for (k = 0; k < CHUNK_LEN; ++k)
					state->output[i][k] += state->tmp[k] / CHUNK_LEN * state->window[k];
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
	state->tmp = fftw_malloc(CHUNK_LEN * sizeof(sample_t));
	memset(state->tmp, 0, CHUNK_LEN * sizeof(sample_t));
	for (i = 0; i < 2; ++i) {
		state->input[i] = fftw_malloc(CHUNK_LEN * sizeof(sample_t));
		memset(state->input[i], 0, CHUNK_LEN * sizeof(sample_t));
		state->in_fr[i] = fftw_malloc(CHUNK_FR_LEN * sizeof(fftw_complex));
		memset(state->in_fr[i], 0, CHUNK_FR_LEN * sizeof(fftw_complex));
		state->r2c_plan[i] = fftw_plan_dft_r2c_1d(CHUNK_LEN, state->tmp, state->in_fr[i], FFTW_ESTIMATE);
	}
	for (i = 0; i < 4; ++i) {
		state->output[i] = fftw_malloc(CHUNK_LEN * sizeof(sample_t));
		memset(state->output[i], 0, CHUNK_LEN * sizeof(sample_t));
		state->out_fr[i] = fftw_malloc(CHUNK_FR_LEN * sizeof(fftw_complex));
		memset(state->out_fr[i], 0, CHUNK_FR_LEN * sizeof(fftw_complex));
		state->c2r_plan[i] = fftw_plan_dft_c2r_1d(CHUNK_LEN, state->out_fr[i], state->tmp, FFTW_ESTIMATE);
	}
	if (e->istream.channels > 2) {
		state->um_buf = calloc(e->istream.channels - 2, sizeof(sample_t *));
		for (i = 0; i < e->istream.channels - 2; ++i)
			state->um_buf[i] = calloc(CHUNK_LEN, sizeof(sample_t));
	}

	return e;
}
