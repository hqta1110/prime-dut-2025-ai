from agent.orchestrator import init_orchestrator

agent = init_orchestrator()
from pprint import pprint
pprint(agent.run("Theo Nghị quyết số 1656/NQ-UBTVQH15, sau khi sắp xếp, sáp nhập, Hà Nội có tổng cộng bao nhiêu đơn vị hành chính cấp xã?").content)
