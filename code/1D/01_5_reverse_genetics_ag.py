import random
import pandas as pd
import numpy as np
import keras
# 1.初始化：随机生成初始种群，每个个体表示一个可能的解。
#
# 2.适应度函数：根据问题的特点，定义适应度函数，用于衡量每个个体的适应性。
#
# 3.选择：根据适应度函数的值，选择适应性较好的个体，作为下一代种群的基础。
#
# 4.交叉：对选出的个体进行交叉操作，产生新的个体。
#
# 5.变异：对新产生的个体进行变异操作，引入新的基因，增加种群多样性。
#
# 6.终止条件：当达到预定的终止条件时，停止算法，并返回最优解。遗传算法具有较好的全局搜索能力和高效的并行性，能够自适应地学习和演化规则
# ，适用于复杂问题的求解。# 但也存在一些问题，如可能陷入局部最优解、需要大量的计算资源和时间等。
# 因此，在使用遗传算法时，需要根据具体问题来选择适当的参数和操作，以获得更好的性能和效果。


CNN_model_y = keras.models.load_model("CNN-model-y-8000.keras")
CNN_model_x = keras.models.load_model("CNN-model-x-8000.keras")

number_draw_picture = 8

Originadata = pd.read_csv('draw-target-%d.txt' % (number_draw_picture), sep='  ',
                          names=['0', '1', '2'], engine='python')
Data = np.array(Originadata)
# print(Data)
XYData = np.array(Data[:, [0, -1]])
# print(XYData)
# print(XYData.shape)


# 1. 初始化种群
## chromosome染色体指的是最小的那个unit cell square方块的长度(长度10*宽带4)，需要将其转化为（100*4）然后作为input参数输入
def initialize_population(population_size, chromosome_length, gene_length):

    population = []
    arrow_population = 0
    while arrow_population < population_size:
        chromosome = []
        arrow_chromosome = 0
        while arrow_chromosome < chromosome_length:
            gene = []
            for __ in range(gene_length):
                gene.append(random.randint(0, 1))
            # for _ in range(20):
            chromosome.append(gene)

            arrow_chromosome += 1

        population.append(chromosome)
        arrow_population += 1

    return population

# 2. 适应度评估
def batch_fitness_functions(population):
    N = len(population)
    full_input = []

    coord = np.array([i / 2 for i in range(200)]).reshape(200, 1)
    normal_coord_xy = (coord - np.mean(coord)) / np.std(coord)

    for individual in population:
        N_Individual = []
        for chromosome in individual:
            N_Individual.extend([chromosome] * 5)  # 复制5次

        New_individual = np.array(N_Individual)  # (200, 2)
        XData_Normal = (New_individual - np.mean(New_individual)) / np.std(New_individual)
        Data_input = np.concatenate((XData_Normal, normal_coord_xy), axis=1)
        Inpdata = np.reshape(Data_input, (200, 3, 1))
        full_input.append(Inpdata)

    full_input = np.array(full_input)  # shape: (N, 200, 3, 1)

    CNN_predict_x_values = np.reshape(CNN_model_x.predict(full_input, verbose=0), (N, 200, 1))  # (N, 200, 1)
    # print(CNN_predict_x_values.shape)
    CNN_predict_y_values = np.reshape(CNN_model_y.predict(full_input, verbose=0), (N, 200, 1))  # (N, 200, 1)

    predict_x_y = np.concatenate([CNN_predict_x_values, CNN_predict_y_values], axis=2)  # (N, 200, 2)
    diff = predict_x_y - XYData  # 广播，shape: (N, 200, 2)

    # 加权误差项
    kx, ky = 1, 2
    constant = np.arange(1, 201)  # shape: (200, )
    constant = constant.reshape(1, 200, 1)  # 广播到每个个体

    square_error = (kx * diff[:, :, 0]**2 + ky * diff[:, :, 1]**2).reshape(N, 200, 1)
    weighted_error = square_error / constant

    rms = np.sqrt(np.mean(weighted_error, axis=1)).flatten()  # shape: (N,)
    return list(rms)



# 3. 选择
def selection(population, fitness_scores):

    population_size = len(population)
    # print(population_size)
    new_fitness_scores = sorted(fitness_scores)

    top_poplulation = []
    top_fitness_scores = []
    size_percent_top = int(population_size * 0.10)
    for individal_scores in new_fitness_scores[:size_percent_top]:
        index_individal = fitness_scores.index(individal_scores)
        top_poplulation.append(population[index_individal])
        top_fitness_scores.append(individal_scores)
    # print(top_poplulation)
    # print(top_fitness_scores)

    second_poplulation = []
    second_fitness_scores = []
    size_percent_second = int(population_size * 0.6)
    for individal_second in new_fitness_scores[size_percent_top:size_percent_second]:
        index_individal_second = fitness_scores.index(individal_second)
        second_poplulation.append(population[index_individal_second])
        second_fitness_scores.append(individal_second)

    total_second_fitness = sum(second_fitness_scores)
    probabilities = [total_second_fitness/score for score in second_fitness_scores]
    second_population_size = len(second_poplulation)
    selected_indices = random.choices(range(second_population_size), weights=probabilities, k=second_population_size)
    updat_second_population = [second_poplulation[i] for i in selected_indices]

    third_poplulation = []
    third_fitness_scores = []
    size_percent_third = int(population_size * 1)
    for individal_third in new_fitness_scores[size_percent_second:size_percent_third]:
        index_individal_third = fitness_scores.index(individal_third)
        third_poplulation.append(population[index_individal_third])
        third_fitness_scores.append(individal_third)

    # total_third_fitness = sum(third_fitness_scores)
    # probabilities = [total_third_fitness / score for score in third_fitness_scores]
    # third_population_size = len(third_poplulation)
    # selected_indices_third = random.choices(range(third_population_size), weights=probabilities, k=third_population_size)
    # updat_third_population = [third_poplulation[i] for i in selected_indices_third]

    return top_poplulation, updat_second_population, third_poplulation

# 4. 交叉

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)

    child1 = [gene.copy() for gene in parent1[:crossover_point]] + [gene.copy() for gene in parent2[crossover_point:]]
    child2 = [gene.copy() for gene in parent2[:crossover_point]] + [gene.copy() for gene in parent1[crossover_point:]]

    return child1, child2


# 5. 变异

def mutation(individual, mutation_rate):
    # 深复制整个个体，避免原始种群被篡改
    new_individual = [gene.copy() for gene in individual]

    for i in range(len(new_individual)):
        if random.random() < mutation_rate:
            # 位翻转：0 -> 1, 1 -> 0
            new_individual[i] = [1 - bit for bit in new_individual[i]]

    return new_individual


# 6. 替换
def replacement(selected_population, offspring, mutation_population):
    return selected_population + offspring + mutation_population

# 7. 遗传算法主循环
def genetic_algorithm(population_size, chromosome_length, gene_length, generations, mutation_rate):
    population = initialize_population(population_size, chromosome_length, gene_length)

    for generation in range(generations):
        # ✅ 批量适应度评估（更高效）
        fitness_scores = batch_fitness_functions(population)

        best_index = np.argmin(fitness_scores)
        best_individual = population[best_index]
        minium_RMS = fitness_scores[best_index]

        print('Generation: %d, Best fitness: %f, population size: %d'
              % (generation + 1, minium_RMS, len(population)))
        print(best_individual)

        if minium_RMS < 0.005:
            break

        # 选择
        top_population, second_population, third_population = selection(population, fitness_scores)

        # 交叉
        offspring = []
        for _ in range(len(second_population) // 2):
            parent1, parent2 = random.sample(second_population, 2)
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # 变异
        mutation_population = [mutation(individual, mutation_rate) for individual in third_population]

        # 替换
        population = replacement(top_population, offspring, mutation_population)

    return best_individual, minium_RMS


# 运行遗传算法


best_solution, minium_RMS = genetic_algorithm(population_size=5000, chromosome_length=40, gene_length=2,
                                              generations=200, mutation_rate=0.3)
print("Best Solution:", best_solution)
print("Best Fitness:", minium_RMS)


Ouputfile = open("best-solution-draw-target-%d.txt" % (number_draw_picture), "w")
for coordinate in best_solution:
    STR_coord = '   '.join([str(coordinate[0]), str(coordinate[1])])
    Ouputfile.write(STR_coord + '\n')

Ouputfile.close()
