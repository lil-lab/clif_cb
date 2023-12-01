#!/usr/bin/env bash
cd ../cb_vin_feedback
exec python -m web_agent.agent_interface ensemble $1 $2 $3
