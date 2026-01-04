import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class EBMOFeatureSelection:
    def __init__(self, n_population=20, max_iter=10, pl_ratio=0.7):
        """
        Enhanced Barnacle Mating Optimization (EBMO).
        :param n_population: Number of barnacles (solutions).
        :param max_iter: Maximum generations.
        :param pl_ratio: mating range parameter (Penis Length ratio).
        """
        self.n = n_population
        self.max_iter = max_iter
        self.pl_ratio = pl_ratio
        
    def fitness(self, X, y, mask):
        """
        Evaluates feature subset using a light wrapper (RF).
        Val loss = 1 - Accuracy.
        """
        if np.sum(mask) == 0:
            return 1.0 # Worst fitness if no features selected
        
        X_sub = X[:, mask.astype(bool)]
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        
        return 1.0 - accuracy_score(y_test, pred)

    def fit(self, X, y):
        """
        Runs EBMO to find best feature mask.
        """
        n_features = X.shape[1]
        
        # Initialization
        population = np.random.randint(0, 2, size=(self.n, n_features))
        best_solution = None
        best_fitness = float('inf')
        
        # Main Loop
        for iteration in range(self.max_iter):
            # Calculate mating range (PL)
            pl = int(self.n * self.pl_ratio)
            
            # Sort population by fitness
            fitness_values = []
            for i in range(self.n):
                f = self.fitness(X, y, population[i])
                fitness_values.append(f)
                
            # Sort barnacles (best to worst)
            sorted_idx = np.argsort(fitness_values)
            population = population[sorted_idx]
            
            if fitness_values[sorted_idx[0]] < best_fitness:
                best_fitness = fitness_values[sorted_idx[0]]
                best_solution = population[0].copy()
                
            print(f"EBMO Iteration {iteration+1}/{self.max_iter}, Best Fitness (Error): {best_fitness:.4f}")

            # Mating & Reproduction
            new_population = []
            # Keep elite (top 10%)
            elite_count = max(2, int(0.1 * self.n))
            new_population.extend(population[:elite_count])
            
            # Generate offspring
            while len(new_population) < self.n:
                # Select parents (Dad and Mum) based on PL constraint
                dad_idx = np.random.randint(0, self.n)
                # Mum must be within PL distance
                start_mum = max(0, dad_idx - pl)
                end_mum = min(self.n, dad_idx + pl)
                mum_idx = np.random.randint(start_mum, end_mum)
                
                dad = population[dad_idx]
                mum = population[mum_idx]
                
                # Crossover (Simple single point or uniform)
                # "Enhanced" aspect: p depends on iteration (adaptive)
                p_cross = 0.5 + 0.4 * (iteration / self.max_iter) 
                
                child = np.where(np.random.rand(n_features) < p_cross, dad, mum)
                
                # Mutation (Exploration)
                mutation_rate = 0.05 * (1 - iteration / self.max_iter) # Decays over time
                if np.random.rand() < mutation_rate:
                     mut_idx = np.random.randint(0, n_features)
                     child[mut_idx] = 1 - child[mut_idx]
                     
                new_population.append(child)
                
            population = np.array(new_population)
            
        self.selected_mask = best_solution
        return best_solution
        
    def transform(self, X):
        return X[:, self.selected_mask.astype(bool)]
