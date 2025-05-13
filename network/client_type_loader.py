import copy


def client_type_loader(pickedClientsList,
                       clientsDatasetDict,
                       networkConfig,
                       basicConfig,
                       typesPerClients,
                       clientConfig,
                       modelToLoad,
                       serverRound, flipboard, turnFlag, sessionId, scorePath, wandbQueue):
    clients = []

    if basicConfig['aggregate_mode'] == 'fedAvg' or basicConfig['aggregate_mode'] == 'fed_avg':
        from client.client_type.client_fedAvg import client_fedAvg

        for i in pickedClientsList:
            clients.append(client_fedAvg(client_internalId=i,
                                         dataset=clientsDatasetDict[i],
                                         networkConfig=networkConfig,
                                         basicConfig=basicConfig,
                                         clientType=typesPerClients[i],
                                         config=clientConfig[int(typesPerClients[i])],
                                         model=copy.deepcopy(modelToLoad),
                                         serverRound=serverRound,
                                         flipboard=flipboard,
                                         turnFlag=turnFlag,
                                         sessionId=sessionId,
                                         scorePath=scorePath,
                                         wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'fed_feature_wise_weighted_avg':
        from client.client_type.client_fedAvg import client_fedAvg

        for i in pickedClientsList:
            clients.append(client_fedAvg(client_internalId=i,
                                         dataset=clientsDatasetDict[i],
                                         networkConfig=networkConfig,
                                         basicConfig=basicConfig,
                                         clientType=typesPerClients[i],
                                         config=clientConfig[int(typesPerClients[i])],
                                         model=copy.deepcopy(modelToLoad),
                                         serverRound=serverRound,
                                         flipboard=flipboard,
                                         turnFlag=turnFlag,
                                         sessionId=sessionId,
                                         scorePath=scorePath,
                                         wandbQueue=wandbQueue))
            
    elif basicConfig['aggregate_mode'] == 'fed_prox':
        from client.client_type.client_fedProx import client_fedProx

        for i in pickedClientsList:
            clients.append(client_fedProx(client_internalId=i,
                                          dataset=clientsDatasetDict[i],
                                          networkConfig=networkConfig,
                                          basicConfig=basicConfig,
                                          clientType=typesPerClients[i],
                                          config=clientConfig[int(typesPerClients[i])],
                                          model=copy.deepcopy(modelToLoad),
                                          serverRound=serverRound,
                                          flipboard=flipboard,
                                          turnFlag=turnFlag,
                                          sessionId=sessionId,
                                          scorePath=scorePath,
                                          wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'partial_fed_prox':
        from client.client_type.client_partial_fedProx import client_partial_fedprox

        for i in pickedClientsList:
            clients.append(client_partial_fedprox(client_internalId=i,
                                                  dataset=clientsDatasetDict[i],
                                                  networkConfig=networkConfig,
                                                  basicConfig=basicConfig,
                                                  clientType=typesPerClients[i],
                                                  config=clientConfig[int(typesPerClients[i])],
                                                  model=copy.deepcopy(modelToLoad),
                                                  serverRound=serverRound,
                                                  flipboard=flipboard,
                                                  turnFlag=turnFlag,
                                                  sessionId=sessionId,
                                                  scorePath=scorePath,
                                                  wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'fisher_server' or basicConfig['aggregate_mode'] == 'fisher_client':
        from client.client_type.client_fisher import client_fisher

        for i in pickedClientsList:
            clients.append(client_fisher(client_internalId=i,
                                         dataset=clientsDatasetDict[i],
                                         networkConfig=networkConfig,
                                         basicConfig=basicConfig,
                                         clientType=typesPerClients[i],
                                         config=clientConfig[int(typesPerClients[i])],
                                         model=copy.deepcopy(modelToLoad),
                                         serverRound=serverRound,
                                         flipboard=flipboard,
                                         turnFlag=turnFlag,
                                         sessionId=sessionId,
                                         scorePath=scorePath,
                                         wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'weighted_fed_avg_param_diff':
        from client.client_type.client_fedAvg import client_fedAvg

        for i in pickedClientsList:
            clients.append(client_fedAvg(client_internalId=i,
                                         dataset=clientsDatasetDict[i],
                                         networkConfig=networkConfig,
                                         basicConfig=basicConfig,
                                         clientType=typesPerClients[i],
                                         config=clientConfig[int(typesPerClients[i])],
                                         model=copy.deepcopy(modelToLoad),
                                         serverRound=serverRound,
                                         flipboard=flipboard,
                                         turnFlag=turnFlag,
                                         sessionId=sessionId,
                                         scorePath=scorePath,
                                         wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'weighted_fed_avg_fisher':
        from client.client_type.client_fisher import client_fisher

        for i in pickedClientsList:
            clients.append(client_fisher(client_internalId=i,
                                         dataset=clientsDatasetDict[i],
                                         networkConfig=networkConfig,
                                         basicConfig=basicConfig,
                                         clientType=typesPerClients[i],
                                         config=clientConfig[int(typesPerClients[i])],
                                         model=copy.deepcopy(modelToLoad),
                                         serverRound=serverRound,
                                         flipboard=flipboard,
                                         turnFlag=turnFlag,
                                         sessionId=sessionId,
                                         scorePath=scorePath,
                                         wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'cka' or basicConfig['aggregate_mode'] == 'fed_cka':
        from client.client_type.client_cka import client_cka

        for i in pickedClientsList:
            clients.append(client_cka(client_internalId=i,
                                      dataset=clientsDatasetDict[i],
                                      networkConfig=networkConfig,
                                      basicConfig=basicConfig,
                                      clientType=typesPerClients[i],
                                      config=clientConfig[int(typesPerClients[i])],
                                      model=copy.deepcopy(modelToLoad),
                                      serverRound=serverRound,
                                      flipboard=flipboard,
                                      turnFlag=turnFlag,
                                      sessionId=sessionId,
                                      scorePath=scorePath,
                                      wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'cosine' or basicConfig['aggregate_mode'] == 'fed_cosine':
        from client.client_type.client_cosine import client_cosine

        for i in pickedClientsList:
            clients.append(client_cosine(client_internalId=i,
                                         dataset=clientsDatasetDict[i],
                                         networkConfig=networkConfig,
                                         basicConfig=basicConfig,
                                         clientType=typesPerClients[i],
                                         config=clientConfig[int(typesPerClients[i])],
                                         model=copy.deepcopy(modelToLoad),
                                         serverRound=serverRound,
                                         flipboard=flipboard,
                                         turnFlag=turnFlag,
                                         sessionId=sessionId,
                                         scorePath=scorePath,
                                         wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'pearson' or basicConfig['aggregate_mode'] == 'fed_pearson':
        from client.client_type.client_pearson import client_pearson

        for i in pickedClientsList:
            clients.append(client_pearson(client_internalId=i,
                                          dataset=clientsDatasetDict[i],
                                          networkConfig=networkConfig,
                                          basicConfig=basicConfig,
                                          clientType=typesPerClients[i],
                                          config=clientConfig[int(typesPerClients[i])],
                                          model=copy.deepcopy(modelToLoad),
                                          serverRound=serverRound,
                                          flipboard=flipboard,
                                          turnFlag=turnFlag,
                                          sessionId=sessionId,
                                          scorePath=scorePath,
                                          wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'l2' or basicConfig['aggregate_mode'] == 'fed_l2':
        from client.client_type.client_l2 import client_l2

        for i in pickedClientsList:
            clients.append(client_l2(client_internalId=i,
                                     dataset=clientsDatasetDict[i],
                                     networkConfig=networkConfig,
                                     basicConfig=basicConfig,
                                     clientType=typesPerClients[i],
                                     config=clientConfig[int(typesPerClients[i])],
                                     model=copy.deepcopy(modelToLoad),
                                     serverRound=serverRound,
                                     flipboard=flipboard,
                                     turnFlag=turnFlag,
                                     sessionId=sessionId,
                                     scorePath=scorePath,
                                     wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'l_inf' or basicConfig['aggregate_mode'] == 'fed_l_inf':
        from client.client_type.client_l_inf import client_l_inf

        for i in pickedClientsList:
            clients.append(client_l_inf(client_internalId=i,
                                        dataset=clientsDatasetDict[i],
                                        networkConfig=networkConfig,
                                        basicConfig=basicConfig,
                                        clientType=typesPerClients[i],
                                        config=clientConfig[int(typesPerClients[i])],
                                        model=copy.deepcopy(modelToLoad),
                                        serverRound=serverRound,
                                        flipboard=flipboard,
                                        turnFlag=turnFlag,
                                        sessionId=sessionId,
                                        scorePath=scorePath,
                                        wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'em' or basicConfig['aggregate_mode'] == 'fed_em':
        from client.client_type.client_em import client_em

        for i in pickedClientsList:
            clients.append(client_em(client_internalId=i,
                                     dataset=clientsDatasetDict[i],
                                     networkConfig=networkConfig,
                                     basicConfig=basicConfig,
                                     clientType=typesPerClients[i],
                                     config=clientConfig[int(typesPerClients[i])],
                                     model=copy.deepcopy(modelToLoad),
                                     serverRound=serverRound,
                                     flipboard=flipboard,
                                     turnFlag=turnFlag,
                                     sessionId=sessionId,
                                     scorePath=scorePath,
                                     wandbQueue=wandbQueue))

    elif basicConfig['aggregate_mode'] == 'fed_2way_avg':
        from client.client_type.client_2way_fed import client_2way_fed

        for i in pickedClientsList:
            clients.append(client_2way_fed(client_internalId=i,
                                          dataset=clientsDatasetDict[i],
                                          networkConfig=networkConfig,
                                          basicConfig=basicConfig,
                                          clientType=typesPerClients[i],
                                          config=clientConfig[int(typesPerClients[i])],
                                          model=copy.deepcopy(modelToLoad),
                                          serverRound=serverRound,
                                          flipboard=flipboard,
                                          turnFlag=turnFlag,
                                          sessionId=sessionId,
                                          scorePath=scorePath,
                                          wandbQueue=wandbQueue))
    
    elif basicConfig['aggregate_mode'] == 'fed_2way_distillation':
        from client.client_type.client_2way_fed_distillation import client_2way_fed_distillation

        for i in pickedClientsList:
            clients.append(client_2way_fed_distillation(client_internalId=i,
                                          dataset=clientsDatasetDict[i],
                                          networkConfig=networkConfig,
                                          basicConfig=basicConfig,
                                          clientType=typesPerClients[i],
                                          config=clientConfig[int(typesPerClients[i])],
                                          model=copy.deepcopy(modelToLoad),
                                          serverRound=serverRound,
                                          flipboard=flipboard,
                                          turnFlag=turnFlag,
                                          sessionId=sessionId,
                                          scorePath=scorePath,
                                          wandbQueue=wandbQueue))
    
    elif basicConfig['aggregate_mode'] == 'fed_2way_ewc':
        from client.client_type.client_2way_ewc import client_2way_ewc

        for i in pickedClientsList:
            clients.append(client_2way_ewc(client_internalId=i,
                                          dataset=clientsDatasetDict[i],
                                          networkConfig=networkConfig,
                                          basicConfig=basicConfig,
                                          clientType=typesPerClients[i],
                                          config=clientConfig[int(typesPerClients[i])],
                                          model=copy.deepcopy(modelToLoad),
                                          serverRound=serverRound,
                                          flipboard=flipboard,
                                          turnFlag=turnFlag,
                                          sessionId=sessionId,
                                          scorePath=scorePath,
                                          wandbQueue=wandbQueue))
    
    elif basicConfig['aggregate_mode'] == 'fed_adaptive_avg':
        from client.client_type.client_adapter_fed import client_adapter_fed

        for i in pickedClientsList:
            clients.append(client_adapter_fed(client_internalId=i,
                                              dataset=clientsDatasetDict[i],
                                              networkConfig=networkConfig,
                                              basicConfig=basicConfig,
                                              clientType=typesPerClients[i],
                                              config=clientConfig[int(typesPerClients[i])],
                                              model=copy.deepcopy(modelToLoad),
                                              serverRound=serverRound,
                                              flipboard=flipboard,
                                              turnFlag=turnFlag,
                                              sessionId=sessionId,
                                              scorePath=scorePath,
                                              wandbQueue=wandbQueue))
    
    elif basicConfig['aggregate_mode'] == 'fed_adaptive_n_guided_avg':
        from client.client_type.client_adapter_fed import client_adapter_fed

        for i in pickedClientsList:
            clients.append(client_adapter_fed(client_internalId=i,
                                              dataset=clientsDatasetDict[i],
                                              networkConfig=networkConfig,
                                              basicConfig=basicConfig,
                                              clientType=typesPerClients[i],
                                              config=clientConfig[int(typesPerClients[i])],
                                              model=copy.deepcopy(modelToLoad),
                                              serverRound=serverRound,
                                              flipboard=flipboard,
                                              turnFlag=turnFlag,
                                              sessionId=sessionId,
                                              scorePath=scorePath,
                                              wandbQueue=wandbQueue))

    else:
        return None

    # print(f'[TEST] {len(clients)}')
    return clients
