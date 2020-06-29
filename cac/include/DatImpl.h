#ifndef CAC_DAT_IMPL_H
#define CAC_DAT_IMPL_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace cac {

class DatImpl {
public:
	mlir::AllocOp allocOp;
};

} // namspace cac


#endif // CAC_DAT_IMPL_H
