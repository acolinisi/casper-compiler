#ifndef CAC_PLATFORM_H
#define CAC_PLATFORM_H

namespace cac {
	class Platform {
	public:
		Platform();
		Platform(unsigned numNodeTypes);
	public:
		unsigned numNodeTypes;
	};
} // namespace cac

#endif // CAC_PLATFORM_H
