#ifndef CAC_MLIR_TYPES_H
#define CAC_MLIR_TYPES_H

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// BaseHalideBuffType
//===----------------------------------------------------------------------===//

/// Base MemRef for Ranked and Unranked variants
class BaseHalideBuffType : public ShapedType {
public:
  using ShapedType::ShapedType;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type) {
    return type.getKind() == CacTypes::HalideBufType;
  }
};

//===----------------------------------------------------------------------===//
// HalideBuffType

/// MemRef types represent a region of memory that have a shape with a fixed
/// number of dimensions. Each shape element can be a non-negative integer or
/// unknown (represented by -1). MemRef types also have an affine map
/// composition, represented as an array AffineMap pointers.
class HalideBuffType : public Type::TypeBase<HalideBuffType, BaseHalideBuffType,
                                         detail::HalideBuffTypeStorage> {
public:
  /// This is a builder type that keeps local references to arguments. Arguments
  /// that are passed into the builder must out-live the builder.
  class Builder {
  public:
    // Build from another HalideBuffType.
    explicit Builder(HalideBuffType other)
        : shape(other.getShape()), elementType(other.getElementType()),
          affineMaps(other.getAffineMaps()),
          memorySpace(other.getMemorySpace()) {}

    // Build from scratch.
    Builder(ArrayRef<int64_t> shape, Type elementType)
        : shape(shape), elementType(elementType), affineMaps(), memorySpace(0) {
    }

    Builder &setShape(ArrayRef<int64_t> newShape) {
      shape = newShape;
      return *this;
    }

    Builder &setElementType(Type newElementType) {
      elementType = newElementType;
      return *this;
    }

    Builder &setAffineMaps(ArrayRef<AffineMap> newAffineMaps) {
      affineMaps = newAffineMaps;
      return *this;
    }

    Builder &setMemorySpace(unsigned newMemorySpace) {
      memorySpace = newMemorySpace;
      return *this;
    }

    operator HalideBuffType() {
      return HalideBuffType::get(shape, elementType, affineMaps, memorySpace);
    }

  private:
    ArrayRef<int64_t> shape;
    Type elementType;
    ArrayRef<AffineMap> affineMaps;
    unsigned memorySpace;
  };

  using Base::Base;

  /// Get or create a new HalideBuffType based on shape, element type, affine
  /// map composition, and memory space.  Assumes the arguments define a
  /// well-formed MemRef type.  Use getChecked to gracefully handle HalideBuffType
  /// construction failures.
  static HalideBuffType get(ArrayRef<int64_t> shape, Type elementType,
                        ArrayRef<AffineMap> affineMapComposition = {},
                        unsigned memorySpace = 0);

  /// Get or create a new HalideBuffType based on shape, element type, affine
  /// map composition, and memory space declared at the given location.
  /// If the location is unknown, the last argument should be an instance of
  /// UnknownLoc.  If the HalideBuffType defined by the arguments would be
  /// ill-formed, emits errors (to the handler registered with the context or to
  /// the error stream) and returns nullptr.
  static HalideBuffType getChecked(ArrayRef<int64_t> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace, Location location);

  ArrayRef<int64_t> getShape() const;

  /// Returns an array of affine map pointers representing the memref affine
  /// map composition.
  ArrayRef<AffineMap> getAffineMaps() const;

  /// Returns the memory space in which data referred to by this memref resides.
  unsigned getMemorySpace() const;

  // TODO(ntv): merge these two special values in a single one used everywhere.
  // Unfortunately, uses of `-1` have crept deep into the codebase now and are
  // hard to track.
  static constexpr int64_t kDynamicSize = -1;
  static int64_t getDynamicStrideOrOffset() {
    return ShapedType::kDynamicStrideOrOffset;
  }

  static bool kindof(unsigned kind) {
    return kind == CacTypes::HalideBuffType;
  }

private:
  /// Get or create a new HalideBuffType defined by the arguments.  If the resulting
  /// type would be ill-formed, return nullptr.  If the location is provided,
  /// emit detailed error messages.
  static HalideBuffType getImpl(ArrayRef<int64_t> shape, Type elementType,
                            ArrayRef<AffineMap> affineMapComposition,
                            unsigned memorySpace, Optional<Location> location);
  using Base::getImpl;
};

} // namespace toy
} // namespace mlir

#endif // CAC_MLIR_TYPES_H
