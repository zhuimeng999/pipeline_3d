# pipeline_3d

现在的三维重建算法一般分为三个步骤，分别是SFM、MVS、FUSE，每个步骤有不同的实现，pipeline_3d允许在每个不同的阶段指定不同的算法，一次调用完成整个重建过程，并包含一个结果评估脚本