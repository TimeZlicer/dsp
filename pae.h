#ifndef _PAE_H
#define _PAE_H

#include "dsp.h"
#include "effect.h"

struct effect * pae_effect_init(struct effect_info *, struct stream_info *, char *, const char *, int, char **);

#endif
