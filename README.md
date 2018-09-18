# -Review-Multi-Agent-Actor-Critic-for-Mixed-Cooperative-Competitive-Environment
[Review] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environment 

> 참고 
> https://jay.tech.blog/2018/08/04/multi-agent-actor-critic-rl/
> 의역과 오역은 언제든지 메일로 조언해주세요
> OpenAI 의 코드는 Fork 한 목록에서 확인해주세요.
> maddpg/maddpg/trainer/maddpg.py & replay_buffer.py 파일을 보면 됩니다.  

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

1. MADDPG 에는 다음과 같은 환경들로 구성이 되어 있습니다.  
  (1) MADDPG에서는 모든 agent들의 움직임을 포함한 전체 state space S  
  (2) 각 agent들의 개별적 observation space O  
  (3) action space A  
  (4) agent별로 Observation과 Action space의 곱으로 표현하는 stochastic policy space   
  (5) Transition space T  

2. critic으로 DQN에서 사용하는 Q-value network를 사용합니다.  
3. Actor는 RL의 Policy Gradient method를 사용하여 학습합니다.  
이때, 적대적(adversarial) 또는 협동적(cooperative) agent들의 개별적인 actor network를 가지되,  
방향성/목표를 유도하기 위해서 centralized critic을 사용 합니다.  
> 즉, centralized 된 critic은 controll agent의 학습 과정에서 other agents의 정보를 critic (Q - value)를 사용하여 학습하게 됩니다.  
> 이 점으로 인해서, P( s' | s ; x, o, policy(1 , ... , N)) = P( s' | s ; x, o, policy'(1 , ... , N)) 과 같아져서 학습에서의 non-stationary 문제를 완화시킬 수 있게 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/45659633-5e512280-bb30-11e8-80af-5220a2b04565.png)

policy gradient 를 통해 actor의 policy 를 구성하는 parameter의 업데이트는 다음과 같은 식을 통해서 이루어 집니다.  

![image](https://user-images.githubusercontent.com/40893452/45659771-fea74700-bb30-11e8-9d11-4d6dc912bf81.png)

Q-value를 보면 state와 모든 agent의 action을 감안한 함수로 표현해 놓은 것을 알 수 있으며, 위 그림에서도 이를 잘 나타내고 있습니다.  
즉 training 과정에서 Q-value를 구하는 과정에서 모든 agent들의 action이 고려된다는 것을 의미하므로 muti-agent가 포함된 환경에서 action-value를 구하고 training하므로 각 agent들은 다른 agent가 포함된 environment에서 policy를 최적화 할 수 있습니다.  
그러므로, 이 action-value function을 centralized action-value function이라고 부르게 됩니다.  

centralizd action-value function은 다음과 같은 loss function을 통해서 업데이트가 이루어집니다.  
![image](https://user-images.githubusercontent.com/40893452/45659847-3f9f5b80-bb31-11e8-8b94-6db1072898d6.png)  

위 식에서, next state에서 다른 agent들의 다음 action을 알고있어야 target Q-value를 구해야 update가 가능하다는 점을 알 수 있습니다.  
즉, 어떤 방법을 통해 다른 agent들의 policy를 알아 낼 수 있다면 위의 식을 그대로 사용할 수 있다는 것을 논문에서도 강조합니다.  
이 논문에서는, 다른 agent들의 policy를 traning중에 알아내는 방법을 사용합니다.  
> 논문에서는 Inferring Policies of Other Agents 라는 Section에서 다룹니다.  

Agent i는 추가로 다른 agent j의 true policy에 대응하는 approximation policy ![image](https://user-images.githubusercontent.com/40893452/45660139-4e3a4280-bb32-11e8-9d00-be5763ff402a.png) 를 지속적으로 추론하도록 합니다.  
online fashion 방법으로 Q function을 위의 수식을 따라 업데이트 하기 이전에, agent j의 replay buffer 에서 sample을 가져와서 ![image](https://user-images.githubusercontent.com/40893452/45660139-4e3a4280-bb32-11e8-9d00-be5763ff402a.png)의 parameter를 업데이트 합니다.  
다른 agent들의 approximation policy는 action의 log probability를 최대화 시키는 방향으로 추론하였으며, entropy regularizer(H)를 도입합니다.  
이와 같이 추론된 policy를 논문에서는 “approximate policy”라고 부릅니다.  

![image](https://user-images.githubusercontent.com/40893452/45660205-95c0ce80-bb32-11e8-9d93-6b0018b4417c.png)

> ![image](https://user-images.githubusercontent.com/40893452/45660415-9efe6b00-bb33-11e8-9db8-ec26f788266d.png)


### [Algorithm]
![image](https://user-images.githubusercontent.com/40893452/45660259-d9b3d380-bb32-11e8-890a-d0861ce26f50.png)


