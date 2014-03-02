#include "gstclassify.h"
#include "recur-common.h"
#include <string.h>

#define PENDING_PROP(self, prop) (&(self)->pending_properties[prop])
#define PP_GET_FLOAT(self, id, def) get_gvalue_float(PENDING_PROP(self, id), def)
#define PP_GET_INT(self, id, def) get_gvalue_int(PENDING_PROP(self, id), def)
#define PP_GET_STRING(self, id, def) get_gvalue_string(PENDING_PROP(self, id), def)
#define PP_GET_BOOLEAN(self, id, def) get_gvalue_boolean(PENDING_PROP(self, id), def)

#define RESET_OR_INIT_GV(v, t) ((G_IS_VALUE(v)) ? g_value_reset(v) : \
      g_value_init((v), (t)))


static inline void
set_gvalue(GValue *dest, const GValue *src)
{
  RESET_OR_INIT_GV(dest, G_VALUE_TYPE(src));
  g_value_copy(src, dest);
}

static inline const char *
get_gvalue_string(GValue *v, const char *_default){
  if (! G_VALUE_HOLDS_STRING(v)){
    return _default;
  }
  return g_value_get_string(v);
}

static inline gboolean
get_gvalue_boolean(GValue *v, const gboolean _default){
  if (! G_VALUE_HOLDS_BOOLEAN(v)){
    return _default;
  }
  return g_value_get_boolean(v);
}

static inline int
get_gvalue_int(GValue *v, const int _default){
  if (! G_VALUE_HOLDS_INT(v)){
    return _default;
  }
  return g_value_get_int(v);
}

static inline u64
get_gvalue_u64(GValue *v, const u64 _default){
  if (! G_VALUE_HOLDS_UINT64(v)){
    return _default;
  }
  return g_value_get_uint64(v);
}

static inline float
get_gvalue_float(const GValue *v, const float _default){
  if (! G_VALUE_HOLDS_FLOAT(v)){
    return _default;
  }
  return g_value_get_float(v);
}

static inline char *
steal_gvalue_string(GValue *v){
  if (! G_VALUE_HOLDS_STRING(v)){
    return NULL;
  }
  char *s = g_value_dup_string(v);
  g_value_unset(v);
  return s;
}

static inline void
set_gvalue_float(GValue *v, const float f){
  RESET_OR_INIT_GV(v, G_TYPE_FLOAT);
  g_value_set_float(v, f);


}
