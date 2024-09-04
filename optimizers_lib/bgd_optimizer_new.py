import torch
from torch.optim.optimizer import Optimizer

            
class BGD_NEW_UPDATE(Optimizer):
    """Implements BGD.
    A simple usage of BGD would be:
    for samples, labels in batches:
        for mc_iter in range(mc_iters):
            optimizer.randomize_weights()
            output = model.forward(samples)
            loss = cirterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.aggaregate_grads()
        optimizer.step()
    """
    def __init__(self, params, server_model_params,  std_init, mean_eta=1, mc_iters=10,alpha_mg = 0.5):
        """
        Initialization of BGD optimizer
        group["mean_param"] is the learned mean.
        group["std_param"] is the learned STD.
        :param params: List of model parameters
        :param std_init: Initialization value for STD parameter
        :param mean_eta: Eta value
        :param mc_iters: Number of Monte Carlo iteration. Used for correctness check.
                         Use None to disable the check.
        """
        super(BGD_NEW_UPDATE, self).__init__(params, defaults={})
        assert mc_iters is None or (type(mc_iters) == int and mc_iters > 0), "mc_iters should be positive int or None."
        self.std_init = std_init
        self.mean_eta = mean_eta
        self.mc_iters = mc_iters
        self.alpha_mg = alpha_mg
        self.server_model_params = {ind:value for ind, (layer_name, value) in enumerate(server_model_params.items())}
        # Initialize mu (mean_param) and sigma (std_param) 
        # breakpoint()
        self.global_model_param_groups = []

        for ind, group in enumerate(self.param_groups):
            assert len(group["params"]) == 1, "BGD optimizer does not support multiple params in a group"
            # group['params'][0] is the weights
            assert isinstance(group["params"][0], torch.Tensor), "BGD expect param to be a tensor"
            # We use the initialization of weights to initialize the mean.
            group["mean_param"] = group["params"][0].data.clone()
            #group["std_param"] = torch.zeros_like(group["params"][0].data).add_(self.std_init)

            group["std_param"] = torch.full_like(group["params"][0].data,self.std_init)

            self.global_model_param_groups.append({"g_mean_param":self.server_model_params[ind]['g_mean_param'].detach().clone(),
                                                   "g_std_param":self.server_model_params[ind]['g_std_param'].detach().clone()})

            # print("Breakpoint 1 Here")
            # breakpoint()

        self._init_accumulators()  

    def get_mc_iters(self):
        return self.mc_iters

    def _init_accumulators(self):
        self.mc_iters_taken = 0
        for group in self.param_groups:
            group["eps"] = None
            group["grad_mul_eps_sum"] = torch.zeros_like(group["params"][0].data)
            group["grad_sum"] = torch.zeros_like(group["params"][0].data)

    def randomize_weights(self, force_std=-1):
        """
        Randomize the weights according to N(mean, std).
        :param force_std: If force_std>=0 then force_std is used for STD instead of the learned STD.
        :return: None
        """
        ## Change here for GMM - 
        for group in self.param_groups:
            mean = group["mean_param"] # theta recieved from the server
            std = group["std_param"] # initialized   
            if force_std >= 0:
                std = std.mul(0).add(force_std)
            group["eps"] = torch.normal(torch.zeros_like(mean), 1)
            # Reparameterization trick (here we set the weights to their randomized value):
            group["params"][0].data.copy_(mean.add(std.mul(group["eps"])))

            '''The above line(Reparamterization) implements this equation -> θi = μi + εiσi'''
    
    def randomize_weights_GMM(self, force_std=-1):
        """ Randomize the weights according to N(mean, std) of the GMM
        group_l: local model parameters
        """
        for (group_l, group_g) in zip(self.param_groups, self.global_model_param_groups):
            mean_l, mean_g = group_l["mean_param"], group_g["g_mean_param"]
            std_l, std_g = group_l["std_param"], group_g["g_std_param"]
            if force_std >= 0:
                std_l = std_l.mul(0).add(force_std)
            mean_gmm = mean_g.mul(self.alpha_mg).add(mean_l.mul(1-self.alpha_mg))
            std_gmm = torch.sqrt(std_l.pow(2).mul(self.alpha_mg).add(std_g.pow(2).mul(1-self.alpha_mg)).add(self.alpha_mg*(1-self.alpha_mg)*(mean_l-mean_g).pow(2)))
            group_l["eps"] = torch.normal(torch.zeros_like(mean_gmm), 1)
            # Reparameterization trick (here we set the weights to their randomized value):
            group_l["params"][0].data.copy_(mean_gmm.add(std_gmm.mul(group_l["eps"])))

            '''The above line(Reparamterization) implements this equation -> θi = μ_gmm^i + εiσ_gmm^i'''

            

    def aggregate_grads(self, batch_size):
        """
        Aggregates a single Monte Carlo iteration gradients. Used in step() for the expectations calculations.
        optimizer.zero_grad() should be used before calling .backward() once again.
        :param batch_size: BGD is using non-normalized gradients, but PyTorch gives normalized gradients.
                            Therefore, we multiply the gradients by the batch size.
        :return: None
        """
        self.mc_iters_taken += 1
        groups_cnt = 0
        for group in self.param_groups:
            if group["params"][0].grad is None:
                continue
            assert group["eps"] is not None, "Must randomize weights before using aggregate_grads"
            groups_cnt += 1
            grad = group["params"][0].grad.data.mul(batch_size)
            group["grad_sum"].add_(grad)
             #The above grad_sum is used to estimate the expectation of gradient which inturn is used in updating μi
            group["grad_mul_eps_sum"].add_(grad.mul(group["eps"]))
            #The above grad_mul_eps_sum is used to estimate the expectation of gradient multiplied by epsilon which inturn is used in updating σi
            group["eps"] = None
        assert groups_cnt > 0, "Called aggregate_grads, but all gradients were None. Make sure you called .backward()"

    def step_nazreen_GMM(self, closure=None):
        """
        Updates the learned mean and STD.
        :return:
        """
        # Makes sure that self.mc_iters had been taken.
        assert self.mc_iters is None or self.mc_iters == self.mc_iters_taken, "MC iters is set to " \
                                                                              + str(self.mc_iters) \
                                                                              + ", but took " + \
                                                                              str(self.mc_iters_taken) + " MC iters"
        # need global mu, sigma
        for ind,group in enumerate(self.param_groups):
            mean = group["mean_param"] # mu k n-1
            std = group["std_param"] # sigma k n-1 

            g_mean = self.global_model_param_groups[ind]["g_mean_param"]
            g_std = self.global_model_param_groups[ind]["g_std_param"]

            var = std.pow(2)
            g_var = g_std.pow(2) 

        
            # Divide gradients by MC iters to get expectation
            e_grad = group["grad_sum"].div(self.mc_iters_taken)
            e_grad_eps = group["grad_mul_eps_sum"].div(self.mc_iters_taken)

            # Update mean and STD params

            # print("Local mean", mean)
            # print("Global mean", g_mean)

            # print("Local std", std)
            # print("Global std", g_std)
            
            denominator = var.mul(self.alpha_mg).add(g_var.mul(1-self.alpha_mg))

            # print("denominator",denominator)

            mean_term1 = var.mul(g_mean).mul(self.alpha_mg).div(denominator)

            # print("mean_term1",mean_term1)

            mean_term2 = g_var.mul(mean).mul(1-self.alpha_mg).div(denominator)

            # print("mean_term2",mean_term2)

            # print("E_grad", e_grad)

            mean_term3 = -(var.mul(g_var).div(denominator)).mul(e_grad).mul(self.mean_eta)

            # print("mean_term3",mean_term3)

            mean.copy_(mean_term1.add(mean_term2).add(mean_term3))

            # print("Final mean", mean)

            sqrt_term = torch.sqrt(e_grad_eps.mul(g_std).mul(std).div(2).pow(2).add(denominator)).mul(g_std.mul(std)).div(denominator)

            # print("sqrt_term", sqrt_term)

            std.copy_(sqrt_term.add(-e_grad_eps.mul(g_std.mul(std).pow(2)).div(denominator.mul(2))))

            # print("Final std", std)

        # self.randomize_weights(force_std=0)
        # To use GMM
        ## ISSUE: The weights are not being randomized properly - Should it be called before updating mean and std (by step func)?
        self.randomize_weights_GMM(force_std=0)
        self._init_accumulators()
    
    def step(self, closure=None):
        """
        Updates the learned mean and STD.
        :return:
        """
        # Makes sure that self.mc_iters had been taken.
        assert self.mc_iters is None or self.mc_iters == self.mc_iters_taken, "MC iters is set to " \
                                                                              + str(self.mc_iters) \
                                                                              + ", but took " + \
                                                                              str(self.mc_iters_taken) + " MC iters"
        self.randomize_weights_GMM(force_std=0)
        # First set the weights to the GMM weights
        for (group_l, group_g) in zip(self.param_groups, self.global_model_param_groups):
            mean_l, mean_g = group_l["mean_param"], group_g["g_mean_param"]
            std_l, std_g = group_l["std_param"], group_g["g_std_param"]
            mean_l.copy_(mean_g.mul(self.alpha_mg).add(mean_l.mul(1-self.alpha_mg)))
            std_l.copy_(torch.sqrt(std_l.pow(2).mul(self.alpha_mg).add(std_g.pow(2).mul(1-self.alpha_mg)).add(self.alpha_mg*(1-self.alpha_mg)*(mean_l-mean_g).pow(2))))
        print(f"Weights are set to GMM weights: last ones are: {mean_l, std_l}")
        for group in self.param_groups:
            mean = group["mean_param"]
            std = group["std_param"]
            # Divide gradients by MC iters to get expectation
            e_grad = group["grad_sum"].div(self.mc_iters_taken)
            e_grad_eps = group["grad_mul_eps_sum"].div(self.mc_iters_taken)
            # Update mean and STD params
            mean.add_(-std.pow(2).mul(e_grad).mul(self.mean_eta))
            sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            std.copy_(sqrt_term.add(-e_grad_eps.mul(std.pow(2)).div(2)))
        self.randomize_weights_GMM(force_std=0)
        self._init_accumulators()