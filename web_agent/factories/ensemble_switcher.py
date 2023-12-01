import subprocess
import yaml

from dataclasses import dataclass

from webapp.agent_factory.abstract_factory import abstract_factory
from webapp.agent_factory import abstract_agent_client
from webapp.utils import args
from typing import Dict


@dataclass
class FactoryArgs:
    agents: Dict[str, str]


with open('../cb_vin_feedback/web_agent/factories/ensemble_switcher.yml'
          ) as infile:
    parsed_yml = yaml.load(infile)
ARGS = FactoryArgs(**parsed_yml)
AGENT_NAMES = sorted(ARGS.agents.keys())


class EnsembleSwitchingFactory(abstract_factory.AbstractFactory):
    def __init__(self):
        super(EnsembleSwitchingFactory, self).__init__()

    def get_new_leader(self, webapp_config: args.WebappArgs):
        pass

    def get_new_follower(
            self, webapp_config: args.WebappArgs,
            agent_idx: int) -> abstract_agent_client.AbstractAgentClient:
        agent_uuid = self.get_random_agent_uuid()

        agent_type = AGENT_NAMES[agent_idx]
        agent_uuid = '%s_%s' % (agent_uuid, agent_type)

        print('Launching an %s agent' % agent_type)

        process = subprocess.Popen([
            'sh', '../cb_vin_feedback/web_agent/launch_ensemble_agent.sh',
            agent_uuid,
            f'../cb_vin_feedback/web_agent/factories/{ARGS.agents[agent_type]}',
            str(webapp_config.port)
        ])

        return abstract_agent_client.SeparateProcessAgentClient(
            agent_uuid, process)
