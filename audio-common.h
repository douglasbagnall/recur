#include "recur-common.h"

/* queue_audio_segment collects up the audio data in a circular buffer, but
   leaves the deinterlacing and interpretation till later. This is a
   decoupling step because the incoming packet size is unrelated to the
   evaluation window size.
 */
static inline void
queue_audio_segment(GstBuffer *buffer, s16 *const queue, const int queue_size,
    int *start, int *end)
{
  GstMapInfo map;
  gst_buffer_map(buffer, &map, GST_MAP_READ);
  int len = map.size / sizeof(s16);

  int lag = *end - *start;
  if (lag < 0){
    lag += queue_size;
  }
  if (lag + len > queue_size){
    GST_WARNING("audio queue lag %d + %d seems to exceed queue size %d",
        lag, len, queue_size);
  }

  if (*end + len < queue_size){
    memcpy(queue + *end, map.data, map.size);
    *end += len;
  }
  else {
    int snip = queue_size - *end;
    int snip8 = snip * sizeof(s16);
    memcpy(queue + *end, map.data, snip8);
    memcpy(queue, map.data + snip8,
        map.size - snip8);
    *end = len - snip;
  }

  GST_LOG("queueing audio starting %" PRIu64  ", ending %" PRIu64,
      GST_BUFFER_PTS(buffer), GST_BUFFER_PTS(buffer) + GST_BUFFER_DURATION(buffer));
  gst_buffer_unmap(buffer, &map);
}
