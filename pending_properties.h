#include "gstclassify.h"
#include "recur-common.h"
#include <string.h>

#define PENDING_PROP(self, prop) (&(self)->pending_properties[prop])

static inline void
set_gvalue(GValue *dest, const GValue *src)
{
  if (G_IS_VALUE(dest)){
    g_value_reset(dest);
  }
  else {
    g_value_init(dest, G_VALUE_TYPE(src));
  }
  g_value_copy(src, dest);
}

static inline const char *
get_gvalue_string(GValue *v){
  if (! G_VALUE_HOLDS_STRING(v)){
    return NULL;
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
