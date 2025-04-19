import gym
from gym import spaces
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self, knn_model):
        super(DroneEnv, self).__init__()

        self.knn = knn_model  # modèle KNN entraîné
        self.state = None

        # 🔵 Espace d'état : 3 features normalisées entre 0 et 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # 🔴 Espace d'action : 3 actions possibles (avance, gauche, droite)
        self.action_space = spaces.Discrete(3)
        
        
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.state = np.random.rand(3).astype(np.float32)
    info = {}  # tu peux ajouter plus d'infos ici si besoin
    return self.state, info


def step(self, action):
        # Modifie l’état en fonction de l’action
        delta = np.zeros(3)
        if action == 0:  # avancer (ex. modifier x)
            delta[0] = 0.05
        elif action == 1:  # gauche (modifier y)
            delta[1] = 0.05
        elif action == 2:  # droite (modifier z)
            delta[2] = 0.05

        self.state = np.clip(self.state + delta + np.random.normal(0, 0.01, 3), 0, 1)

        # Calcul de la récompense avec le modèle KNN
        predicted_error = self.knn.predict(self.state.reshape(1, -1))[0]
        reward = -predicted_error  # Plus l'erreur est grande, plus la récompense est négative

        done = False
        info = {}
        terminated = False
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

def render(self, mode='human'):
        print(f"Drone state: {self.state}")