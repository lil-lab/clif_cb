import subprocess
import yaml

from dataclasses import dataclass

from webapp.agent_factory.abstract_factory import abstract_factory
from webapp.agent_factory import abstract_agent_client
from webapp.utils import args


@dataclass
class AgentArgs:
    experiment_name: str = ''
    model_save: str = ''
    use_sampling: str = 'False'


class InstructionFollowerFactory(abstract_factory.AbstractFactory):
    def __init__(self):
        super(InstructionFollowerFactory, self).__init__()
        pass

    def get_new_leader(self, webapp_config: args.WebappArgs):
        pass

    def get_new_follower(
        self, webapp_config: args.WebappArgs
    ) -> abstract_agent_client.AbstractAgentClient:
        agent_uuid = self.get_random_agent_uuid()
        print('Launching agent process!')
        config_path = webapp_config.agent_config_path

        with open(config_path, 'r') as infile:
            parsed_yml = yaml.load(infile)
        agent_args = AgentArgs(**parsed_yml)

        if agent_args.use_sampling not in {'True', 'False'}:
            raise ValueError(
                'Use sampling should be either True or False (string value)')

        if not agent_args.experiment_name:
            raise ValueError('No experiment name set!')
        if not agent_args.model_save:
            raise ValueError('No model save set!')

        process = subprocess.Popen([
            'sh', '../cb_vin_feedback/web_agent/launch_agent.sh', agent_uuid,
            agent_args.experiment_name, agent_args.model_save,
            str(webapp_config.port), agent_args.use_sampling
        ])
        return abstract_agent_client.SeparateProcessAgentClient(
            agent_uuid, process)
