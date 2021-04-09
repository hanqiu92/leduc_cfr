# Counterfactual Regret Minimization（CFR）的简单实现

本仓库中各文件分别包括以下内容：

* cfr.py: 实现了vanilla CFR算法。
* cfr_p.py: 实现了CFR+算法。
* mccfr_external.py: 实现了MCCFR算法的一种基础变种：external-sampling MCCFR。
* mccfr_outcome.py: 实现了MCCFR算法的另一种基础变种：outcome-sampling MCCFR。
* deep_cfr.py: 实现了一种基础的Deep CFR算法。
* leduc.py: 实现了一个Leduc游戏环境，可以通过一组超参数调节游戏难度。
* util.py: 包括一些评估策略效果的函数。
