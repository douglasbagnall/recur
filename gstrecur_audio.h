#ifndef __GST_AUDIO_RECUR_AUDIO_H__
#define __GST_AUDIO_RECUR_AUDIO_H__

#include <gst/audio/gstaudiofilter.h>
#include <gst/audio/audio.h>
#include "recur-context.h"
#include "recur-common.h"

G_BEGIN_DECLS
#define GST_TYPE_RECUR_AUDIO (gst_recur_audio_get_type())
#define GST_RECUR_AUDIO(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_RECUR_AUDIO,GstRecurAudio))
#define GST_RECUR_AUDIO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_RECUR_AUDIO,GstRecurAudioClass))
#define GST_IS_RECUR_AUDIO(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_RECUR_AUDIO))
#define GST_IS_RECUR_AUDIO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_RECUR_AUDIO))


typedef struct _GstRecurAudio GstRecurAudio;
typedef struct _GstRecurAudioClass GstRecurAudioClass;

struct _GstRecurAudio
{
  GstAudioFilter audiofilter;
  RecurContext *context;
};


struct _GstRecurAudioClass
{
  GstAudioFilterClass parent_class;
};

GType gst_recur_audio_get_type(void);

void gst_recur_audio_register_context (GstRecurAudio * self, RecurContext *context);


G_END_DECLS
#endif /* __GST_AUDIO_RECUR_AUDIO_H__ */
