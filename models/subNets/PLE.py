import torch
import torch.nn as nn
import numpy as np

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, p=0.1):
        super(Expert, self).__init__()

        self.dnn_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(p)
        )
        

    def forward(self, x):
        out = self.dnn_layer(x)
        return out
    
class Expert1(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, p=0.1):
        super(Expert1, self).__init__()

        self.dnn_layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p)
        )
        

    def forward(self, x):
        out = self.dnn_layer(x)
        return out
    
class PLElayer(nn.Module):

    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, num_gates):
        super(PLElayer, self).__init__()

        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.num_gates = num_gates # 3表示低层特征提取

        self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])

        # gates
        self.gate_task1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                 nn.Softmax(dim=1))
        self.gate_task2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                 nn.Softmax(dim=1))
        if self.num_gates == 3:
            self.gate_shared = nn.Sequential(nn.Linear(self.input_size, 2 * self.num_specific_experts + self.num_shared_experts),
                                 nn.Softmax(dim=1))
            
    def forward(self, x_1, x_s, x_2):

        experts_task1_o = [e(x_1) for e in self.experts_task1]
        experts_task1_o = torch.cat(([e[:, np.newaxis, :] for e in experts_task1_o]), dim=1)

        experts_shared_o = [e(x_s) for e in self.experts_shared]
        experts_shared_o = torch.cat(([e[:, np.newaxis, :] for e in experts_shared_o]), dim=1)

        experts_task2_o = [e(x_2) for e in self.experts_task2]
        experts_task2_o = torch.cat(([e[:, np.newaxis, :] for e in experts_task2_o]), dim=1)

        # gate 1
        gate_1 = self.gate_task1(x_1)
        gate_1 = gate_1.unsqueeze(2)
        gate_expert_output1 = torch.cat([experts_task1_o, experts_shared_o], dim=1)
        gate_task1_out = torch.matmul(gate_expert_output1.transpose(1, 2), gate_1)
        gate_task1_out = gate_task1_out.squeeze(2)

        # gate 2
        gate_2 = self.gate_task2(x_2)
        gate_2 = gate_2.unsqueeze(2)
        gate_expert_output2 = torch.cat([experts_task2_o, experts_shared_o], dim=1)
        gate_task2_out = torch.matmul(gate_expert_output2.transpose(1, 2), gate_2)
        gate_task2_out = gate_task2_out.squeeze(2)

        if self.num_gates == 3:
            gate_s = self.gate_shared(x_s)
            gate_s = gate_s.unsqueeze(2)
            gate_expert_output3 = torch.cat([experts_task1_o, experts_shared_o, experts_task2_o], dim=1)
            gate_shared_out = torch.matmul(gate_expert_output3.transpose(1, 2), gate_s)
            gate_shared_out = gate_shared_out.squeeze(2)
            return gate_task1_out, gate_shared_out, gate_task2_out

        return gate_task1_out, gate_task2_out

        
    # def forward(self, x_1, x_s, x_2):

    #     experts_task1_o = [e(x_1) for e in self.experts_task1]
    #     experts_task1_o = torch.stack(experts_task1_o)

    #     experts_shared_o = [e(x_s) for e in self.experts_shared]
    #     experts_shared_o = torch.stack(experts_shared_o)
        
    #     experts_task2_o = [e(x_2) for e in self.experts_task2]
    #     experts_task2_o = torch.stack(experts_task2_o)

    #     # gate1
    #     selected1 = self.gate_task1(x_1)
    #     gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
    #     gate_task1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        
    #     # gate2
    #     selected2 = self.gate_task2(x_2)
    #     gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
    #     gate_task2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        
    #     # shared gate
    #     if self.num_gates == 3:
    #         selected3 = self.gate_shared(x_s)
    #         gate_expert_output3 = torch.cat((experts_task1_o, experts_shared_o, experts_task2_o), dim=0)
    #         gate_shared_out = torch.einsum('abc, ba -> bc', gate_expert_output3, selected3)
    #         return gate_task1_out, gate_shared_out, gate_task2_out

    #     return gate_task1_out, gate_task2_out 
    
class PLE(nn.Module):

    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden):
        super(PLE, self).__init__()
        self.extraction_layer = PLElayer(input_size, num_specific_experts, num_shared_experts, 
                                         experts_out, experts_hidden, num_gates=3)
        self.cgc = PLElayer(input_size, num_specific_experts, num_shared_experts, 
                            experts_out, experts_hidden, num_gates=2)
        
    
    def forward(self, x):

        output_task1, output_shared, output_task2 = self.extraction_layer(x, x, x)
        output_task1, output_task2 = self.cgc(output_task1, output_shared, output_task2)

        return output_task1, output_task2
    
class PLE1(nn.Module):

    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden):
        super(PLE1, self).__init__()
        # self.extraction_layer = PLElayer(input_size, num_specific_experts, num_shared_experts, 
        #                                  experts_out, experts_hidden, num_gates=3)
        self.cgc = PLElayer(input_size, num_specific_experts, num_shared_experts, 
                            experts_out, experts_hidden, num_gates=2)
        
    
    def forward(self, x):

        # output_task1, output_shared, output_task2 = self.extraction_layer(x, x, x)
        output_task1, output_task2 = self.cgc(x, x, x)

        return output_task1, output_task2


if __name__=='__main__':

    x = torch.randn(3, 768)
    m = PLElayer(768, 2, 2, 768, 768, 3)
    o = m(x, x, x)

        


