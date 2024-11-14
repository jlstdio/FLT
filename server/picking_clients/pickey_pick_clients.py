def pickeyPickClients(self):
    N = self.serverConfig['dont_pick_recent_classes']

    # 최근 N 라운드의 클래스 집합을 생성
    with self.recentPickedClasses_lock:
        recent_classes = set().union(*self.recentPickedClasses)

    print('recent_classes:', recent_classes)

    # 클라이언트가 eligible한지 검사하는 함수
    def is_eligible(client):
        return not self.totalDistributionSet.get(client, set()).intersection(recent_classes)

    eligible_clients = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 클라이언트 목록을 병렬로 검사
        results = executor.map(is_eligible, self.clientsList)
        eligible_clients = [client for client, eligible in zip(self.clientsList, results) if eligible]

    # 클라이언트 선택 로직은 기존과 동일
    if len(eligible_clients) >= self.updateClientsPerRound:
        pickedClients = np.random.choice(eligible_clients, self.updateClientsPerRound, replace=False)
    else:
        print('not enough eligible clients')
        pickedClients = list(eligible_clients)
        remaining = self.updateClientsPerRound - len(eligible_clients)

        if remaining > 0:
            additional_clients = list(set(self.clientsList) - set(eligible_clients))

            if len(additional_clients) >= remaining:
                pickedClients += list(np.random.choice(additional_clients, remaining, replace=False))
            else:
                pickedClients += additional_clients

        pickedClients = np.array(pickedClients)

    # 클라이언트 상태 업데이트
    for session_id, client_id in enumerate(pickedClients):
        self.sessionId[client_id] = session_id
        self.status[client_id] = False
        self.turnFlag[client_id] = 1  # mark the client which is picked
        self.flipboard[client_id] = 0  # mark as file not sent

    for i in range(self.updateClientsPerRound):
        self.pickedClientsList[i] = pickedClients[i]

    # 현재 라운드의 클래스 정보를 수집하여 recentPickedClasses를 업데이트
    current_round_classes = list(
        set().union(*(self.totalDistributionSet.get(client, set()) for client in pickedClients))
    )

    with self.recentPickedClasses_lock:
        self.recentPickedClasses.append(current_round_classes)
        if len(self.recentPickedClasses) > N + 1:
            self.recentPickedClasses.pop(0)