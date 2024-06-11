import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    print(len(trainloaders), len(trainloaders[0].dataset))

    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.0,
    #     min_fit_clients=cfg.num_clients_per_round_fit,
    #     fraction_evaluate=0.0,
    #     min_evaluate_clients=cfg.num_clients_per_round_eval,
    #     min_available_clients=cfg.num_clients,
    #     on_fit_config_fn=get_on_fit_config(cfg.config_fit),
    #     evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    # )

    strategy = fl.server.strategy.FedProx(
        fraction_fit=0.0,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.0,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        proximal_mu=0.1  # Add the proximal term parameter
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
