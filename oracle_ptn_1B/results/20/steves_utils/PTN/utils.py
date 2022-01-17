#! /usr/bin/env python3

import torch

from steves_models.steves_ptn import Steves_Prototypical_Network



def independent_prediction(model:Steves_Prototypical_Network, episode):
    model.eval()
    with torch.no_grad():
        """
        Keep in mind that y in this case are pseudo labels, in that they range from [0,n_way), they are not the original labels
        shapes:
            support_x: n_way*n_shot,<x shape>
            support_y: n_way*n_shot
            query_x: n_way*n_query,<x shape>
            query_y: n_way*n_query
        """
        support_x, support_y, query_x, query_y, real_classes = episode


        # for j in [support_x, support_y, query_x, query_y]: print(j.shape)

        # Shape n_way*n_shot,<backbone output>
        support_z = model.backbone.forward(support_x.cuda())

        # print(support_z.shape)

        """
        Compute the prototypes for each pseudo_y by averaging the z vectors for each label
        shapes:
            support_prototypes: n_way, <backbone output>
        """
        n_way = len(torch.unique(support_y))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        support_prototypes = torch.cat(
            [
                support_z[torch.nonzero(support_y == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # print(support_prototypes.shape)

        """
        Compute the distance/score for each query_x
        For each query_z, its the distance to each support_prototype
        Basically this ends up looks like the probability coming out of a regular network
        shapes:
            query_z: n_way*n_query, <backbone output>
            dists: n_way*n_query, n_way
        """
        query_z = model.backbone.forward(query_x.cuda())
        # print(query_z.shape)

        # Compute the  2-norm (IE Euclidean distance) from each query to each prototype
        dists = torch.cdist(query_z, support_prototypes)
        # print(dists.shape)

        # Negative distance is the "score"
        scores = -dists


        """
        Compute the loss
        Again, for each query_x we get a distance to each support_prototype, so it looks similar to the output of a regular CNN
        shapes:
            loss: [] (is a scalar)
        """
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(scores, query_y.cuda())

        """
        Choose the least distant
        shapes:
            y_hat: n_way*n_query
            num_correct: [] (scalar)
        """
        # something_1 = (== query_y.cuda()).sum().item()
        # something_1 = ([1].cuda()== query_y.cuda()).sum().item()
        y_hat = torch.max(scores,dim=1,)[1].cuda() # We get the index of the highest prototype score for each query_x
        n_correct = (y_hat.cuda() == query_y.cuda()).sum().item()
        n_total = query_y.shape[0]


        return y_hat, n_correct, n_total, loss




def independent_accuracy_assesment(model:Steves_Prototypical_Network, dl):
    correct = 0
    total   = 0

    for episode in dl:
        y_hat, n_correct, n_total, loss = independent_prediction(model, episode)

        correct += n_correct
        total += n_total
    
    return correct/total



if __name__ == "__main__":
    import numpy as np
    a = np.zeros([])