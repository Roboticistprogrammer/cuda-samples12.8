//  (C) Copyright John Maddock 2001 - 2003.
//  (C) Copyright Darin Adler 2001.
//  (C) Copyright Jens Maurer 2001 - 2003.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for most recent version.

//  generic SGI STL:

#if !defined(__STL_CONFIG_H)
#include <boost/config/no_tr1/utility.hpp>
#if !defined(__STL_CONFIG_H)
#error "This is not the SGI STL!"
#endif
#endif

//
// No std::iterator traits without partial specialisation:
//
#if !defined(__STL_CLASS_PARTIAL_SPECIALIZATION)
#define BOOST_NO_STD_ITERATOR_TRAITS
#endif

//
// No std::stringstream with gcc < 3
//
#if defined(__GNUC__) && (__GNUC__ < 3) && ((__GNUC_MINOR__ < 95) || (__GNUC_MINOR__ == 96)) \
        && !defined(__STL_USE_NEW_IOSTREAMS)                                                 \
    || defined(__APPLE_CC__)
// Note that we only set this for GNU C++ prior to 2.95 since the
// latest patches for that release do contain a minimal <sstream>
// If you are running a 2.95 release prior to 2.95.3 then this will need
// setting, but there is no way to detect that automatically (other
// than by running the configure script).
// Also, the unofficial GNU C++ 2.96 included in RedHat 7.1 doesn't
// have <sstream>.
#define BOOST_NO_STRINGSTREAM
#endif

//
// Assume no std::locale without own iostreams (this may be an
// incorrect assumption in some cases):
//
#if !defined(__SGI_STL_OWN_IOSTREAMS) && !defined(__STL_USE_NEW_IOSTREAMS)
#define BOOST_NO_STD_LOCALE
#endif

//
// Original native SGI streams have non-standard std::messages facet:
//
#if defined(__sgi) && (_COMPILER_VERSION <= 650) && !defined(__SGI_STL_OWN_IOSTREAMS)
#define BOOST_NO_STD_LOCALE
#endif

//
// SGI's new iostreams have missing "const" in messages<>::open
//
#if defined(__sgi) && (_COMPILER_VERSION <= 740) && defined(__STL_USE_NEW_IOSTREAMS)
#define BOOST_NO_STD_MESSAGES
#endif

//
// No template iterator constructors, or std::allocator
// without member templates:
//
#if !defined(__STL_MEMBER_TEMPLATES)
#define BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
#define BOOST_NO_STD_ALLOCATOR
#endif

//
// We always have SGI style hash_set, hash_map, and slist:
//
#define BOOST_HAS_HASH
#define BOOST_HAS_SLIST

//
// If this is GNU libstdc++2, then no <limits> and no std::wstring:
//
#if (defined(__GNUC__) && (__GNUC__ < 3))
#include <string>
#if defined(__BASTRING__)
#define BOOST_NO_LIMITS
// Note: <boost/limits.hpp> will provide compile-time constants
#undef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#define BOOST_NO_STD_WSTRING
#endif
#endif

//
// There is no standard iterator unless we have namespace support:
//
#if !defined(__STL_USE_NAMESPACES)
#define BOOST_NO_STD_ITERATOR
#endif

//
// Intrinsic type_traits support.
// The SGI STL has it's own __type_traits class, which
// has intrinsic compiler support with SGI's compilers.
// Whatever map SGI style type traits to boost equivalents:
//
#define BOOST_HAS_SGI_TYPE_TRAITS

//  C++0x headers not yet implemented
//
#define BOOST_NO_0X_HDR_ARRAY
#define BOOST_NO_0X_HDR_CHRONO
#define BOOST_NO_0X_HDR_CODECVT
#define BOOST_NO_0X_HDR_CONCEPTS
#define BOOST_NO_0X_HDR_CONDITION_VARIABLE
#define BOOST_NO_0X_HDR_CONTAINER_CONCEPTS
#define BOOST_NO_0X_HDR_FORWARD_LIST
#define BOOST_NO_0X_HDR_FUTURE
#define BOOST_NO_0X_HDR_INITIALIZER_LIST
#define BOOST_NO_0X_HDR_ITERATOR_CONCEPTS
#define BOOST_NO_0X_HDR_MEMORY_CONCEPTS
#define BOOST_NO_0X_HDR_MUTEX
#define BOOST_NO_0X_HDR_RANDOM
#define BOOST_NO_0X_HDR_RATIO
#define BOOST_NO_0X_HDR_REGEX
#define BOOST_NO_0X_HDR_SYSTEM_ERROR
#define BOOST_NO_0X_HDR_THREAD
#define BOOST_NO_0X_HDR_TUPLE
#define BOOST_NO_0X_HDR_TYPE_TRAITS
#define BOOST_NO_STD_UNORDERED // deprecated; see following
#define BOOST_NO_0X_HDR_UNORDERED_MAP
#define BOOST_NO_0X_HDR_UNORDERED_SET

#define BOOST_STDLIB "SGI standard library"
