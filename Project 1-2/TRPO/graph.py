# episode = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440,
#            460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
#            900, 920, 940, 960, 980]
# loss = [2.7755575615628914e-19, 3.5195451900980064e-17, -7.978700512607245e-17, -8.881784197001253e-18,
#         8.585724723767877e-17, -6.135947631779794e-17, -7.84557604068444e-17, 3.549411900288604e-17,
#         -6.030459325958371e-18, -1.564231752908296e-17, 6.839948067617509e-17, -1.1887665273233034e-17,
#         1.4574356307820586e-17, -7.898819938044593e-18, -4.957274900651862e-18, 2.365213422525586e-17,
#         -4.244119294831181e-18, -3.467599863957898e-17, 7.062223168803615e-18, -2.4671622769447922e-17,
#         -6.6809881127885525e-18, -1.2724619193411537e-17, -1.879511419641013e-17, 6.543554172643359e-18,
#         -1.6406318415955295e-17, -1.9308226515220114e-18, -1.6436877562196635e-17, 2.352517189726168e-17,
#         -9.156478553609538e-18, 1.327994380306627e-16, 2.149512148354611e-17, -1.8085080944120197e-17,
#         -1.1859949224482384e-17, 2.2437300402338722e-17, 2.7334327419101663e-17, -2.413528314402514e-18,
#         -1.36702922996941e-17, -1.2583880717871975e-17, -2.8675917564489206e-17, -2.4245088799821437e-18,
#         9.654113257610057e-18, 1.2389585627900614e-17, -1.239224236086634e-16, -7.733277619802814e-17,
#         -3.317192977404763e-18, -7.716580536056691e-19, 3.66641369913317e-17, -1.1407796216331884e-17,
#         9.578394722256253e-17, 5.3304031189804964e-17]
# reward = [0.0007095567332129349, 0.03629128446316924, 0.1422871862196355, 0.10568708335803764, 0.0008726205644222773, 0.17881659524956592, 0.07146717129639829, 0.6243614154204532, 0.11792301504570007, 0.00569592743748866, 0.04210059460160434, 0.0768900848868986, 0.1279281346598374, 0.1631399447067862, 0.25270796407086155, 0.26408339730756586, 0.24402098178596798, 0.19768616881997073, 0.1985661450297975, 0.39921634439263914, 0.2934227745285897, 0.3657697325778246, 0.3837570401283233, 0.38332776769798504, 0.5255661506135795, 0.4609419217466536, 0.9697054170125606, 0.4669427246524812, 0.9003552987007245, 1.1222141004361255, 0.3473389882526295, 0.32269278902064563, 0.4941417190530189, 0.41055609692246825, 0.39559544072834985, 0.3955577641238256, 0.6483265781190131, 0.4007318622356187, 0.950354162002172, 0.7771901496222278, 0.8392670311846968, 0.701028434026239, 0.9423143405575791, 0.8621555004755294, 0.7579598835855444, 1.13491081167147, 0.7227839330177043, 0.8026539623856099, 1.2392585876614515, 1.7738093135152304]

# reward [3.001208245754242, 94.09595936536789, 44.21892386674881, 204.66985285282135, 5903.932458418916, 942.7436018060835, 31.47733300924301, 101.90738385915756, 3855.497513187074, 149.83263927698135, 8774.759054835828, 6058.5485274158455, 1671.7016017138958, 8378.00673500539, 9270.055458893505, 272.1910760104656, 3284.6568648173707, 8968.083700193463, 197.77486194372176, 694.0310854315758, 432.8017997741699, 8644.257166162635, 8639.115672037413, 269.39589956486014, 8747.561290789177, 3174.029433497184, 432.9626352004998, 2341.436509155319, 9774.53798742086, 9737.420284576947, 9701.725302952982, 1593.3590570307279, 9716.45182886136, 9742.892735723151, 9659.954125955694, 9639.654347009819, 9683.240821269632, 9796.697429740932, 9761.178190864099, 9668.528770185301, 1017.1328312982805, 9590.316138496433, 8246.801969573775, 9847.577888863801, 9712.510130718056, 9791.75972652185, 9956.519982733662, 9801.954887643235, 1076.34461494704, 9662.139787902568]


import matplotlib.pyplot as plt
def zhexiantu(episode, reward, batch_size, gamma):
    x = episode
    y1 = reward
    # y2 = random_reward

    plt.plot(x, y1, ms=10)
    # plt.plot(x, y2, ms=10, label="Random Action Reward")

    # plt.xticks(rotation=45)
    plt.xlabel("Episode")
    plt.ylabel("Expected Reward")
    plt.title("Reward for BallBalancer robot when batchsize={}, gamma={}".format(batch_size,gamma))
    # plt.legend(loc="upper left")
    # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    # for y in [y1, y2, y3, y4]:
    #     for x1, yy in zip(x, y):
    #         plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=20, rotation=0)
    plt.savefig("storage/BallBalancer_batchsize{}_gamma{}.jpg".format(batch_size,gamma))
    plt.show()

def parazhexian(episode, gamma, reward, batch_size):
    x = episode
    y = reward
    # y2 = random_reward
    for i in range(len(gamma)):
        plt.plot(x, y[i], ms=10,label=("Batchsize={}".format(gamma[i])))
    # plt.plot(x, y2, ms=10, label="Random Action Reward")

    # plt.xticks(rotation=45)
    plt.xlabel("Episode")
    plt.ylabel("Expected Reward")
    plt.title("Reward for CartpoleSwing for different batchsize when gamma={}".format(batch_size))
    plt.legend(loc="upper left")
    # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    # for y in [y1, y2, y3, y4]:
    #     for x1, yy in zip(x, y):
    #         plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=20, rotation=0)
    plt.savefig("storage/Cartpole_Swing_gamma{}.jpg".format(batch_size))
    plt.show()
