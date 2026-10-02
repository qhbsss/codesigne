# 本轮优化分析

历史起点：final-optimized-30394-20261002 的官方 grade 合格，分数 30394.21557，P1 611770，D1 71412，面积 23.9357，峰值 16.936/18.0744 W。
目标乘积 P1*D1 <= 16143631757.536（分数 >=50000）。

重要：原 P1 两个 batch 独立 barrier，而 resident2/4 使用共同 barrier。孤立阶段不能直接比较总 bytes 或运行时间，原孤立阶段只包含一个 batch。

候选：
1. m32：扩大 MMA M 到32，保持 RF 合法；功能通过。单 batch W1 从54765到51353，约6.2%。
2. 8x8 k4：面积23.538，D1精评69184，峰值18.0645；只改善3.1%。
3. resident2：每 batch16组，每SM2组，prompt各阶段全局对齐，QKV/attention分两半，W2 4Kx4N。W1两batch阶段44440，W2两batch31808。发现并修复常规gemm第二acc面板占用lane7与epilogue临时lane7冲突；新功能检查中。
4. fused8：FFN激活保持RF中，W1/GELU/W2融合；W2分8个K段，再归并。测试中。
5. resident4：每batch32组，每SM4组，QKV按query块拆分，W2 4Kx8N；测试中。

所有优化仅改变生成器/硬件与输出程序，未改公开评分器。
