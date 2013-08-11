/*
 * This file is originally from Xiph.org's libvorbis. For license terms, see
 * COPYING.mdct. (BSDish).
 *
 * COPYRIGHT 1994-2009 Xiph.Org Foundation http://www.xiph.org/
 *
 * lightly modified by Douglas Bagnall <douglas@halo.gen.nz>
 */

#ifndef _V_WINDOW_
#define _V_WINDOW_

extern float *_vorbis_window_get(int n);
extern void _vorbis_apply_window(float *d,int *winno,long *blocksizes,
                          int lW,int W,int nW);


#endif
