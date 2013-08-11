#ifndef __GST_MANAGER_RECUR_MANAGER_H__
#define __GST_MANAGER_RECUR_MANAGER_H__

#include <gst/gst.h>
#include "recur-context.h"

G_BEGIN_DECLS
#define GST_TYPE_RECUR_MANAGER \
  (gst_recur_manager_get_type())
#define GST_RECUR_MANAGER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_RECUR_MANAGER,GstRecurManager))
#define GST_RECUR_MANAGER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_RECUR_MANAGER,GstRecurManagerClass))
#define GST_IS_RECUR_MANAGER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_RECUR_MANAGER))
#define GST_IS_RECUR_MANAGER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_RECUR_MANAGER))

//

typedef struct _GstRecurManager GstRecurManager;
typedef struct _GstRecurManagerClass GstRecurManagerClass;

struct _GstRecurManager
{
  GstBin parent;
  RecurContext *context;
};


struct _GstRecurManagerClass
{
  GstBinClass parent_class;
};

GType gst_recur_manager_get_type(void);

G_END_DECLS
#endif /* __GST_MANAGER_RECUR_MANAGER_H__ */
