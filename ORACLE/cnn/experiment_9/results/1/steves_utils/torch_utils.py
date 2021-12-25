from re import U
import torch
import math

def split_dataset_by_percentage(train:float, val:float, test:float, dataset, seed:int):
    assert train < 1.0
    assert val < 1.0
    assert test < 1.0
    assert train + val + test <= 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = math.floor(len(dataset) * test)

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))



"""
Assumes batch is in the form (X,Y,U) or (X,Y)
"""
def predict_batch(model, device, batch, forward_uses_domain):
    model.eval()
    tup = [t.to(device) for t in batch]

    if forward_uses_domain:
        y_hat, u_hat = model.forward(tup[0], tup[2])
    else:
        y_hat = model.forward(tup[0])
    pred = y_hat.data.max(1, keepdim=True)[1]
    pred = torch.flatten(pred).cpu()

    return pred


"""
Returns dict in the form of
{
    <domain>:
    {
        <y>: {
            <y_hat>: int
        }
    },
}

IE [domain][y][y_hat]:count

IE a confusion matrix for each domain
"""
def confusion_by_domain_over_dataloader(model, device, dl, forward_uses_domain, denormalize_domain_func=None):
    confusion_by_domain = {}

    for batch in dl:
        pred = predict_batch(model, device, batch, forward_uses_domain)

        batch.append(pred)



        for x,y,u,y_hat in zip(*batch):
            x = x.cpu().detach().numpy()
            y = int(y.cpu().detach().numpy())
            u = float(u.cpu().detach().numpy())
            y_hat = int(y_hat.cpu().detach().numpy())

            if denormalize_domain_func is not None:
                u = denormalize_domain_func(u)

            u = int(u)

            # Yeah yeah I know...
            if u not in confusion_by_domain:
                confusion_by_domain[u] = {}
            if y not in confusion_by_domain[u]:
                confusion_by_domain[u][y] = {}
            if y_hat not in confusion_by_domain[u][y]:
                confusion_by_domain[u][y][y_hat] = 0

            confusion_by_domain[u][y][y_hat] += 1

    return confusion_by_domain


def ptn_confusion_by_domain_over_dataloader(model, device, dl):
    confusion_by_domain = {}
    correct_and_total_by_domain = {} # Going to use this to validate

    for u, (support_x, support_y, query_x, query_y, classes) in dl:
        # Returns pseudo labels so we need to fetch them from the classes list that 
        # is generated for each episode
        pseudo_y_hat = model.predict_on_one_task(
            support_x,
            support_y,
            query_x,
            query_y,
        )

        y_hat = [classes[idx] for idx in pseudo_y_hat]

        # These two chunks are just for sanity checking our confusion matrix
        correct, total, _loss = model.evaluate_on_one_task(
            support_x, support_y, query_x, query_y
        )

        if u not in correct_and_total_by_domain:
            correct_and_total_by_domain[u] = [0, 0]
        correct_and_total_by_domain[u][0] += correct
        correct_and_total_by_domain[u][1] += total


         
        for y, y_hat in zip(query_y, y_hat):
            y = int(y.detach().item())
            # Yeah yeah I know...
            if u not in confusion_by_domain:
                confusion_by_domain[u] = {}
            if y not in confusion_by_domain[u]:
                confusion_by_domain[u][y] = {}
            if y_hat not in confusion_by_domain[u][y]:
                confusion_by_domain[u][y][y_hat] = 0
            confusion_by_domain[u][y][y_hat] += 1

    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(confusion_by_domain)

    # ok now check the matrix by domain
    for domain, item_1 in confusion_by_domain.items():
        n_correct = 0
        n_total = 0
        for y, y_hat_dict in item_1.items():
            if y in y_hat_dict: # Handle the case if we never had a single correct guess
                n_correct += y_hat_dict[y]
            n_total += sum(y_hat_dict.values())
        
        assert(
            correct_and_total_by_domain[domain][0] == n_correct
        )
        assert(
            correct_and_total_by_domain[domain][1] == n_total
        )
        


    return confusion_by_domain