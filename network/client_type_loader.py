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

    if basicConfig['aggregate_mode'] == 'fedAvg':
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

    elif basicConfig['aggregate_mode'] == 'fed_fisher_server' or basicConfig['aggregate_mode'] == 'fed_fisher_client':
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
    else:
        return None

    return clients
