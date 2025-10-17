
#ifndef AXON_EXPORT_H
#define AXON_EXPORT_H

#ifdef AXON_STATIC_DEFINE
#  define AXON_EXPORT
#  define AXON_NO_EXPORT
#else
#  ifndef AXON_EXPORT
#    ifdef axon_EXPORTS
        /* We are building this library */
#      define AXON_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define AXON_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef AXON_NO_EXPORT
#    define AXON_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef AXON_DEPRECATED
#  define AXON_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef AXON_DEPRECATED_EXPORT
#  define AXON_DEPRECATED_EXPORT AXON_EXPORT AXON_DEPRECATED
#endif

#ifndef AXON_DEPRECATED_NO_EXPORT
#  define AXON_DEPRECATED_NO_EXPORT AXON_NO_EXPORT AXON_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef AXON_NO_DEPRECATED
#    define AXON_NO_DEPRECATED
#  endif
#endif

#endif /* AXON_EXPORT_H */
