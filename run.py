from agent import *

agent = init_orchestrator()
format_agent = init_format_agent()

answer = agent.run("Yếu tố nào là cốt lõi giúp công nghiệp văn hóa tạo ra giá trị kinh tế?\nA. Sự sáng tạo và đổi mới nội dung\nB. Sử dụng tài nguyên thiên nhiên\nC. Giảm giá thành sản xuất\nD. Tăng cường lao động phổ thông").content
formatted_answer = format_agent.run(answer).to_dict()['content']
print(formatted_answer)