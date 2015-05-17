#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "effect.h"
#include "util.h"

#include "biquad.h"
#include "gain.h"
#include "crossfeed.h"
#include "remix.h"
#include "delay.h"
#include "resample.h"
#include "fir.h"
#include "noise.h"
#include "compress.h"
#include "stats.h"

static struct effect_info effects[] = {
	{ "lowpass_1",          "lowpass_1 f0[k]",                       biquad_effect_init },
	{ "highpass_1",         "highpass_1 f0[k]",                      biquad_effect_init },
	{ "lowpass",            "lowpass f0[k] width[q|o|h|k]",          biquad_effect_init },
	{ "highpass",           "highpass f0[k] width[q|o|h|k]",         biquad_effect_init },
	{ "bandpass_skirt",     "bandpass_skirt f0[k] width[q|o|h|k]",   biquad_effect_init },
	{ "bandpass_peak",      "bandpass_peak f0[k] width[q|o|h|k]",    biquad_effect_init },
	{ "notch",              "notch f0[k] width[q|o|h|k]",            biquad_effect_init },
	{ "allpass",            "allpass f0[k] width[q|o|h|k]",          biquad_effect_init },
	{ "eq",                 "eq f0[k] width[q|o|h|k] gain",          biquad_effect_init },
	{ "lowshelf",           "lowshelf f0[k] width[q|s|o|h|k] gain",  biquad_effect_init },
	{ "highshelf",          "highshelf f0[k] width[q|s|o|h|k] gain", biquad_effect_init },
	{ "linkwitz_transform", "linkwitz_transform fz[k] qz fp[k] qp",  biquad_effect_init },
	{ "deemph",             "deemph",                                biquad_effect_init },
	{ "biquad",             "biquad b0 b1 b2 a0 a1 a2",              biquad_effect_init },
	{ "gain",               "gain [channel] gain",                   gain_effect_init },
	{ "mult",               "mult [channel] multiplier",             gain_effect_init },
	{ "crossfeed",          "crossfeed f0[k] separation",            crossfeed_effect_init },
	{ "remix",              "remix channel_selector|. ...",          remix_effect_init },
	{ "delay",              "delay seconds",                         delay_effect_init },
#ifdef __HAVE_FFTW3__
#ifndef __SYMMETRIC_IO__
	{ "resample",           "resample [bandwidth] fs",               resample_effect_init },
#endif
	{ "fir",                "fir impulse_file",                      fir_effect_init },
#endif
	{ "noise",              "noise level",                           noise_effect_init },
	{ "compress",           "compress thresh ratio attack release",  compress_effect_init },
	{ "stats",              "stats [ref_level]",                     stats_effect_init },
};

struct effect_info * get_effect_info(const char *name)
{
	int i;
	for (i = 0; i < LENGTH(effects); ++i)
		if (strcmp(name, effects[i].name) == 0)
			return &effects[i];
	return NULL;
}

struct effect * init_effect(struct effect_info *ei, struct stream_info *istream, char *channel_selector, int argc, char **argv)
{
	return ei->init(ei, istream, channel_selector, argc, argv);
}

void destroy_effect(struct effect *e)
{
	e->destroy(e);
	free(e);
}

void append_effect(struct effects_chain *chain, struct effect *e)
{
	if (chain->tail == NULL)
		chain->head = e;
	else
		chain->tail->next = e;
	chain->tail = e;
	e->next = NULL;
}

int build_effects_chain(int argc, char **argv, struct effects_chain *chain, struct stream_info *stream, char *channel_selector, const char *dir)
{
	int i = 1, k = 0, j, last_selector_index = -1, old_stream_channels;
	char *tmp_channel_selector;
	struct effect_info *ei = NULL;
	struct effect *e = NULL;

	if (channel_selector == NULL) {
		channel_selector = NEW_SELECTOR(stream->channels);
		SET_SELECTOR(channel_selector, stream->channels);
	}
	else {
		tmp_channel_selector = NEW_SELECTOR(stream->channels);
		COPY_SELECTOR(tmp_channel_selector, channel_selector, stream->channels);
		channel_selector = tmp_channel_selector;
	}

	while (k < argc) {
		if (argv[k][0] == ':') {
			if (parse_selector(&argv[k][1], channel_selector, stream->channels))
				goto fail;
			last_selector_index = k++;
			i = k + 1;
			continue;
		}
		if (argv[k][0] == '@') {
			old_stream_channels = stream->channels;
			if (build_effects_chain_from_file(chain, stream, channel_selector, dir, &argv[k][1]))
				goto fail;
			if (stream->channels != old_stream_channels) {
				tmp_channel_selector = NEW_SELECTOR(stream->channels);
				if (last_selector_index == -1)
					SET_SELECTOR(tmp_channel_selector, stream->channels);
				else if (parse_selector(&argv[last_selector_index][1], tmp_channel_selector, stream->channels)) {
					free(tmp_channel_selector);
					goto fail;
				}
				free(channel_selector);
				channel_selector = tmp_channel_selector;
			}
			i = ++k + 1;
			continue;
		}
		ei = get_effect_info(argv[k]);
		if (ei == NULL) {
			LOG(LL_ERROR, "dsp: error: no such effect: %s\n", argv[k]);
			goto fail;
		}
		while (i < argc && get_effect_info(argv[i]) == NULL && argv[i][0] != ':' && argv[i][0] != '@')
			++i;
		if (LOGLEVEL(LL_VERBOSE)) {
			fprintf(stderr, "dsp: effect:");
			for (j = 0; j < i - k; ++j)
				fprintf(stderr, " %s", argv[k + j]);
			fprintf(stderr, "; channels=%d [", stream->channels);
			print_selector(channel_selector, stream->channels);
			fprintf(stderr, "] fs=%d\n", stream->fs);
		}
		e = init_effect(ei, stream, channel_selector, i - k, &argv[k]);
		if (e == NULL) {
			LOG(LL_ERROR, "dsp: error: failed to initialize effect: %s\n", argv[k]);
			goto fail;
		}
		append_effect(chain, e);
		k = i;
		i = k + 1;
		if (e->ostream.channels != stream->channels) {
			tmp_channel_selector = NEW_SELECTOR(e->ostream.channels);
			if (last_selector_index == -1)
				SET_SELECTOR(tmp_channel_selector, e->ostream.channels);
			else if (parse_selector(&argv[last_selector_index][1], tmp_channel_selector, e->ostream.channels)) {
				free(tmp_channel_selector);
				goto fail;
			}
			free(channel_selector);
			channel_selector = tmp_channel_selector;
		}
		*stream = e->ostream;
	}
	free(channel_selector);
	return 0;

	fail:
	free(channel_selector);
	return 1;
}

int build_effects_chain_from_file(struct effects_chain *chain, struct stream_info *stream, char *channel_selector, const char *dir, const char *path)
{
	char **argv = NULL, *env, *tmp = NULL, *d = NULL, *p = NULL, *c = NULL;
	int i, argc = 0;

	if (path[0] != '\0' && path[0] == '~' && path[1] == '/') {
		env = getenv("HOME");
		i = strlen(env) + strlen(&path[1]) + 1;
		p = calloc(i, sizeof(char));
		snprintf(p, i, "%s%s", env, &path[1]);
	}
	else if (dir == NULL || path[0] == '/')
		p = strdup(path);
	else {
		i = strlen(dir) + 1 + strlen(path) + 1;
		p = calloc(i, sizeof(char));
		snprintf(p, i, "%s/%s", dir, path);
	}
	if (!(c = get_file_contents(p))) {
		LOG(LL_ERROR, "dsp: info: failed to load effects file: %s: %s\n", p, strerror(errno));
		goto fail;
	}
	if (gen_argv_from_string(c, &argc, &argv))
		goto fail;
	d = strdup(p);
	tmp = strrchr(d, '/');
	if (tmp == NULL)
		d = strdup(".");
	else
		*tmp = '\0';
	LOG(LL_VERBOSE, "dsp: info: begin effects file: %s\n", p);
	if (build_effects_chain(argc, argv, chain, stream, channel_selector, d))
		goto fail;
	LOG(LL_VERBOSE, "dsp: info: end effects file: %s\n", p);
	free(c);
	free(p);
	free(d);
	for (i = 0; i < argc; ++i)
		free(argv[i]);
	free(argv);
	return 0;

	fail:
	free(c);
	free(p);
	for (i = 0; i < argc; ++i)
		free(argv[i]);
	free(argv);
	return 1;
}

double get_effects_chain_max_ratio(struct effects_chain *chain)
{
	struct effect *e = chain->head;
	double ratio = 1.0, max_ratio = 1.0;
	while (e != NULL) {
		ratio *= e->worst_case_ratio;
		max_ratio = (ratio > max_ratio) ? ratio : max_ratio;
		e = e->next;
	}
	return max_ratio;
}

double get_effects_chain_total_ratio(struct effects_chain *chain)
{
	struct effect *e = chain->head;
	double ratio = 1.0;
	while (e != NULL) {
		ratio *= e->ratio;
		e = e->next;
	}
	return ratio;
}

sample_t * run_effects_chain(struct effects_chain *chain, ssize_t *frames, sample_t *buf1, sample_t *buf2)
{
	sample_t *ibuf = buf1, *obuf = buf2, *tmp;
	struct effect *e = chain->head;
	while (e != NULL && *frames > 0) {
		e->run(e, frames, ibuf, obuf);
		tmp = ibuf;
		ibuf = obuf;
		obuf = tmp;
		e = e->next;
	}
	return ibuf;
}

void reset_effects_chain(struct effects_chain *chain)
{
	struct effect *e = chain->head;
	while (e != NULL) {
		e->reset(e);
		e = e->next;
	}
}

void plot_effects_chain(struct effects_chain *chain, int input_fs)
{
	int i = 0, k, j, max_fs = -1, channels = -1;
	struct effect *e = chain->head;
	while (e != NULL) {
		if (e->plot == NULL) {
			LOG(LL_ERROR, "dsp: plot: error: effect '%s' does not support plotting\n", e->name);
			return;
		}
		if (channels == -1)
			channels = e->ostream.channels;
		else if (channels != e->ostream.channels) {
			LOG(LL_ERROR, "dsp: plot: error: number of channels cannot change: %s\n", e->name);
			return;
		}
		e = e->next;
	}
	printf(
		"set xlabel 'frequency (Hz)'\n"
		"set ylabel 'amplitude (dB)'\n"
		"set logscale x\n"
		"set samples 500\n"
		"set grid xtics ytics\n"
		"set key on\n"
	);
	e = chain->head;
	while (e != NULL) {
		e->plot(e, i++);
		max_fs = (e->ostream.fs > max_fs) ? e->ostream.fs : max_fs;
		e = e->next;
	}
	for (k = 0; k < channels; ++k) {
		printf("Hsum%d(f)=", k);
		for (j = 0; j < i; ++j)
			printf("H%d_%d(f) + ", k, j);
		if (k == 0)
			printf("0\nplot [f=10:%d/2] [-30:20] Hsum%d(f) title 'Channel %d'\n", (max_fs == -1) ? input_fs : max_fs, k, k);
		else
			printf("0\nreplot Hsum%d(f) title 'Channel %d'\n", k, k);
	}
}

sample_t * drain_effects_chain(struct effects_chain *chain, ssize_t *frames, sample_t *buf1, sample_t *buf2)
{
	ssize_t dframes = -1;
	sample_t *ibuf = buf1, *obuf = buf2, *tmp;
	double fratio = 1.0;
	struct effect *e = chain->head;
	while (e != NULL && dframes == -1) {
		dframes = *frames * fratio;
		e->drain(e, &dframes, ibuf);
		fratio *= e->ratio * e->istream.channels / e->ostream.channels;
		e = e->next;
	}
	*frames = dframes;
	while (e != NULL && *frames > 0) {
		e->run(e, frames, ibuf, obuf);
		tmp = ibuf;
		ibuf = obuf;
		obuf = tmp;
		e = e->next;
	}
	return ibuf;
}

void destroy_effects_chain(struct effects_chain *chain)
{
	struct effect *e;
	while (chain->head != NULL) {
		e = chain->head;
		chain->head = e->next;
		destroy_effect(e);
	}
	chain->tail = NULL;
}

void print_all_effects(void)
{
	int i;
	fprintf(stdout, "Effects:\n");
	for (i = 0; i < LENGTH(effects); ++i)
		fprintf(stdout, "  %s\n", effects[i].usage);
}
