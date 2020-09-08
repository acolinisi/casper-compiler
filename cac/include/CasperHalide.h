#pragma once

#include <Halide.h>

namespace cac {

// TODO: handle Tunable for any type (at least any Arithmetic
// template type)? dynamic_cast method used now won't work with a
// template type, need to add an myType() method that returns
// an enum to base class (GeneratorParamBase)
#if 0
template<typename T>
class TunableGeneratorParam : public Halide::GeneratorParam<T> {
    TunableGeneratorParam(const std::string &name, const T &value,
			const T &min = std::numeric_limits<T>::lowest(),
			const T &max = std::numeric_limits<T>::max())
        : Halide::GeneratorParam<T>(name, value, min, max) {}
};
#else
class TunableGeneratorParam : public Halide::GeneratorParam<int> {
public:
    TunableGeneratorParam(const std::string &name, const int &value,
			const int &min = std::numeric_limits<int>::lowest(),
			const int &max = std::numeric_limits<int>::max())
        : Halide::GeneratorParam<int>(name, value, min, max) {}
};
class InputGeneratorParam : public Halide::GeneratorParam<int> {
public:
    InputGeneratorParam(const std::string &name, const int &value,
			const int &min = std::numeric_limits<int>::lowest(),
			const int &max = std::numeric_limits<int>::max())
        : Halide::GeneratorParam<int>(name, value, min, max) {}
};
#endif

} // namespace cac
