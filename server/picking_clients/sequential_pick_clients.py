from server.picking_clients.picker_parent import picker_parent


class sequential_pick_clients(picker_parent):
    def __init__(self, initial_data):
        super().__init__(initial_data)

    def pick_clients(self):
        pair_size = self.initial_data['pair_size']
        total_clients = self.initial_data['total_clients']
        curRound = self.initial_data['curRound']
        set_length = int(total_clients / pair_size)

        # round          | 1  2  3  4  5  6  7  8  9  10 ...
        # adjusted round | 0  1  2  3  4  5  6  7  8  9  ...
        # rest           | 0  1  2  3  4  0  1  2  3  4  ...
        # set_to_pick    | 1  2  3  4  5  1  2  3  4  5  ...

        set_to_pick = ((curRound - 1) % set_length) + 1

        start_idx = (set_to_pick - 1) * pair_size
        end_idx = (set_to_pick * pair_size)

        pickedClients = []
        for i in range(start_idx, end_idx):
            pickedClients.append(i)

        return pickedClients
