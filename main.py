from graph import run_agent, logger
import time
import json
import asyncio
from graph import setup_agent

def format_task_status(status):
    if status == "completed":
        return "✅ 已完成"
    elif status == "in_progress":
        return "🔄 进行中"
    elif status == "failed":
        return "❌ 失败"
    else:
        return "⏳ 等待中"

async def main():
    print("\n" + "="*50)
    print("欢迎使用LangGraph工作流Agent！")
    print("="*50)
    print("请输入您的指令，Agent将会思考并执行任务。")
    print("输入'exit'退出程序。")
    print("="*50 + "\n")
    
    # 初始化Agent和工具
    print("🔧 正在初始化工具和工作流...")
    try:
        # 提前初始化agent和工具，这样后续交互时不需要再初始化
        agent_workflow = await setup_agent()
        print("✅ 初始化完成！可用工具已加载。")
        
        # 提示用户使用示例
        print("\n💡 您可以尝试这些示例指令：")
        print("  - 计算(15 * 8) / 2 + 10的结果")
        print("  - 搜索关于人工智能最新进展的信息")
        print("  - 打开一个网页并截图")
        print("\n开始使用吧！")
    except Exception as e:
        logger.error(f"初始化过程中发生错误: {str(e)}", exc_info=True)
        print(f"⚠️ 初始化时遇到问题: {str(e)}")
        print("系统将使用基本工具继续运行。某些高级功能可能不可用。")
        print("如果问题持续存在，请检查MCP服务是否正常运行。")
    
    while True:
        try:
            user_input = input("\n📝 请输入您的指令: ")
            
            if user_input.lower() in ('exit', 'quit', 'q', 'bye'):
                print("\n👋 感谢使用，再见！")
                break
                
            if not user_input.strip():
                print("⚠️ 请输入有效指令")
                continue
                
            print("\n🔄 正在处理您的请求...\n")
            start_time = time.time()
            
            # 运行agent (异步)
            final_state = await run_agent(user_input)
            end_time = time.time()
            
            # 打印最终结果
            print("\n" + "="*50)
            print(f"✨ 执行结果 (耗时: {end_time - start_time:.2f}秒)")
            print("="*50)
            
            # 打印任务列表及其状态
            tasks = final_state.get("tasks", [])
            if tasks:
                print("\n📋 任务列表:")
                for i, task in enumerate(tasks):
                    status_text = format_task_status(task["status"])
                    tool_info = f" 🔧 工具: {task.get('tool', 'unknown')}" if 'tool' in task else ""
                    print(f"{i+1}. [{status_text}] {task['description']}{tool_info}")
                    if task.get("result"):
                        result_text = task.get("result")
                        # 如果结果文本太长，截断显示
                        if len(result_text) > 100:
                            result_text = result_text[:100] + "..."
                        print(f"   📌 结果: {result_text}")
            
            # 打印最终消息
            messages = final_state.get("messages", [])
            if messages:
                # 获取最后一条消息（人性化回复）
                last_message = messages[-1] if messages else None
                if last_message and hasattr(last_message, "content"):
                    print("\n💬 最终回应:")
                    print("-" * 50)
                    content = last_message.content
                    print(f"{content}")
                    print("-" * 50)
            
            print("\n" + "="*50)
            print("✅ 处理完成！")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 操作已中断")
            user_continue = input("是否要退出程序？(y/n): ")
            if user_continue.lower() in ('y', 'yes'):
                print("\n👋 感谢使用，再见！")
                break
        except Exception as e:
            logger.error(f"执行过程中发生错误: {str(e)}", exc_info=True)
            print(f"\n❌ 执行过程中发生错误: {str(e)}")
            print("请尝试输入其他指令或重启程序。")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
