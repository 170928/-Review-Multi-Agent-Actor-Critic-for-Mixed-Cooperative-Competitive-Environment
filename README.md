# -Review-Multi-Agent-Actor-Critic-for-Mixed-Cooperative-Competitive-Environment
[Review] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environment 

> 의역과 오역은 언제든지 메일로 조언해주세요

> About paper  
> Ryan Lowe, yi Wu, Aviv Tamar, Jean harb, Pieter Abbeel, Igor Mordatch   
> OpenAI & UC Berkeley   

### [Motivation]
1. RL 이 Game (Atari, Baduik 등) 뿐만 아니라 Robotics (공정, 데이터 센터 쿨링 등) 의 분야 까지 효과적인 문제 해결 능력을 보여주고 있습니다. 그러나 대부분의 경우 "single agent"의 해결 방법을 제시하고 있습니다.  

2. 현실적인 문제들 (multi-robot control, multiplayer game 등)은 multiple agents 들이 서로 상호작용하는 환경의 문제가 더 많이 발생하였고, hierarchical RL, self-play 등의 방법을 통해서 시도가 있어왔습니다.  

3. 기존의 RL (Q-Learning, policy gradient 등) 의 방법들은 multiple agent 환경에서 각각의 agent가 가지는 policy가 훈련과정 중 지속적으로 바뀌는 현상에 의해서 여러가지 문제 ( non-stationary, high variance ) 가 발생하는 것을 효과적으로 해결하지 못하였습니다.  

4. Competitive & Cooperative 환경에서 모두 같은 구조로써 학습할 수 있는 General multi agent learning algorithm에 대한 필요성이 대두되고 있습니다.  

### [Property of Proposed Model]
General purpose multi-agent learning algorithm :  
(1) learned policy는 실행 시간 (execution time)에 local information 만을 필요로 합니다.  
> 학습 (Training) 과정에서는 다른 agent의 Q-value를 공유하여 학습하나, 학습이 끝난 후 실행시에는 agent가 관찰한 정보만을 사용합니다.  

(2) agent들 간에 communication을 위햇 터그별한 구조를 가정하지 않습니다.  
(3) cooperative & competitive 환경에서 모두 가ㅌ은 구조를 통해서 학습하고 실행될 수 있습니다.  

### [Methodology]
(1) Centralized training with decentralized execution : 학습 과정에서는 다른 agent의 정보를 활용하기 위해서 centrailized training이 수행되며, 실행 시에는 agent가 관찰하는 local information만을 사용하므로 decetralized execution이 가능해진다.   
(2) extension of actor-critic policy gradient method 기반의 알고리즘을 제안하며, 기존의 Q-learning이 학습 / 실행 과정에서 다른 정보를 사용하는 것이 어려운 점을 해결 합니다.  
(3) centralized critic funciton은 다른 agent의 decision-making policy를 사용하기 때문에, 학습 과정에서 효과적으로 상대의 정보를 인지할 수 있습니다.  

### [MADDPG]
> Multi-agent deep deterministic policy gradient


