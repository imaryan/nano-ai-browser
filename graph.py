from typing import Annotated, Dict, List, Literal, Optional, TypedDict, Union
import operator
import logging
import json
import asyncio
import re  # 添加正则表达式模块导入
from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, Tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

# 使用 OpenAI 替代 Anthropic Claude
from langchain_openai import ChatOpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_workflow.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("agent_workflow")

# 全局MCP客户端
mcp_client = None


# 定义状态类型
class TaskItem(TypedDict):
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    result: NotRequired[str]
    tool: NotRequired[str]
    tool_name: NotRequired[str]
    tool_args: NotRequired[str]
    task_type: NotRequired[Literal["动作执行", "信息检索", "任务检查"]]
    snapshot: NotRequired[str]
    extracted_data: NotRequired[str]
    last_execution: NotRequired[bool]
    tools_call: NotRequired[List[Dict]]  # 存储工具调用数组


class TaskManager(TypedDict):
    tasks: List[TaskItem]
    current_task_index: Optional[int]
    

class AgentState(MessagesState):
    task_manager: TaskManager  # 结构化的任务管理器
    execution_status: Literal["thinking", "executing", "finished"]
    direct_to_end: NotRequired[bool]  # 是否直接结束工作流的标记


# 初始化LLM - 设置为异步模式
model = ChatOpenAI(
    model="qwen-plus",
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=False,  # 异步模式下通常不使用流式输出
)

# 用于任务处理的独立模型实例 - 避免上下文累积
task_model = ChatOpenAI(
    model="qwen-plus",
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=False,
)


# 获取MCP工具
async def get_mcp_tool():
    """从MCP客户端获取工具"""
    global mcp_client

    try:
        logger.info("连接MCP客户端...")
        # 创建MCP客户端（如果尚未创建）
        if mcp_client is None:
            mcp_client = MultiServerMCPClient(
                {
                    "playwright": {
                        "transport": "sse",
                        "url": "http://127.0.0.1:8931/sse",
                    }
                }
            )
            # 初始化客户端
            await mcp_client.connect_to_server(
                server_name="playwright",
                transport="sse",
                url="http://127.0.0.1:8931/sse",
            )
            logger.info("MCP客户端连接成功")

        # 获取工具
        mcp_tools = mcp_client.get_tools()
        if not mcp_tools:
            logger.warning("未获取到MCP工具")
            return []

        logger.info(f"获取到 {len(mcp_tools)} 个MCP工具")

        return mcp_tools
    except Exception as e:
        logger.error(f"获取MCP工具时出错: {str(e)}")
        # 如果出错，重置客户端以便下次重新连接
        mcp_client = None
        return []


# 在程序结束时关闭MCP客户端
async def close_mcp_client():
    global mcp_client
    if mcp_client:
        try:
            await mcp_client.exit_stack.aclose()
            logger.info("MCP客户端已关闭")
        except Exception as e:
            logger.error(f"关闭MCP客户端时出错: {str(e)}")
        finally:
            mcp_client = None


# 模拟工具执行函数 - 异步版本
async def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    logger.info(f"执行网络搜索: {query}")
    # 模拟网络延迟
    await asyncio.sleep(0.5)
    return f"搜索结果: 关于 '{query}' 的信息"


async def calculate(expression: str) -> str:
    """执行数学计算"""
    logger.info(f"执行数学计算: {expression}")
    try:
        # 计算操作通常很快，但为了一致性也设为异步
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        logger.error(f"计算错误: {str(e)}")
        return "无法计算该表达式"


# 定义异步工具包装函数
def create_async_tool(func, name, description):
    """创建一个异步工具对象"""

    async def _run(query: str):
        return await func(query)

    return Tool(
        name=name,
        description=description,
        func=_run,
        coroutine=func,  # 设置协程函数
    )


# 初始化工具节点函数 - 异步
async def initialize_tool_node():
    # 直接获取MCP工具
    logger.info("直接获取MCP工具...")
    try:
        # 获取MCP工具
        mcp_tools = await get_mcp_tool()

        # 如果没有获取到MCP工具，使用基本工具作为备用
        if not mcp_tools:
            logger.warning("未获取到MCP工具，将使用基本工具")
            # 创建基本工具列表作为备用
            base_tools = [
                create_async_tool(
                    func=search_web,
                    name="search_web",
                    description="搜索互联网获取信息",
                ),
                create_async_tool(
                    func=calculate,
                    name="calculate",
                    description="执行数学计算",
                ),
            ]
            tools_list = base_tools
        else:
            logger.info(f"成功获取 {len(mcp_tools)} 个MCP工具，将只使用MCP工具")
            # 直接使用MCP工具，不再添加模拟工具
            tools_list = mcp_tools
            logger.info(f"工具列表只包含 {len(tools_list)} 个MCP工具")
    except Exception as e:
        logger.error(f"获取MCP工具时出错: {str(e)}")
        logger.warning("将使用基本工具")
        # 创建基本工具列表作为备用
        base_tools = [
            create_async_tool(
                func=search_web,
                name="search_web",
                description="搜索互联网获取信息",
            ),
            create_async_tool(
                func=calculate,
                name="calculate",
                description="执行数学计算",
            ),
        ]
        tools_list = base_tools

    # 创建工具节点
    return ToolNode(tools_list), tools_list


# 运行工具节点初始化
async def setup_agent():
    global tool_node, tools
    logger.info("设置Agent和工具...")
    tool_node, tools = await initialize_tool_node()
    logger.info(f"已初始化工具节点，包含 {len(tools)} 个工具")

    # 构建工作流
    workflow = build_agent_workflow()
    logger.info("工作流已构建")
    return workflow


# 在模块加载时不立即初始化工具列表，等到真正需要时再初始化
tool_node = None
tools = []

# 程序结束时清理资源
import atexit
import asyncio


def cleanup_resources():
    """清理程序使用的资源"""
    logger.info("程序结束，开始清理资源...")
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(close_mcp_client())
            else:
                loop.run_until_complete(close_mcp_client())
        except RuntimeError as e:
            logger.warning(f"清理资源时无法获取事件循环: {str(e)}")
            logger.info("MCP客户端将在程序终止时自动关闭")
            global mcp_client
            mcp_client = None
        logger.info("资源清理完成")
    except Exception as e:
        logger.error(f"清理资源时出错: {str(e)}")


atexit.register(cleanup_resources)


# 合并的工具执行节点 - 包含工具调用和结果处理
async def tool_execution(state: AgentState) -> AgentState:
    logger.info("=== 进入工具执行节点 ===")
    
    messages = state["messages"]
    task_manager = state.get("task_manager", {"tasks": [], "current_task_index": None})
    tasks = task_manager.get("tasks", [])
    current_task_index = task_manager.get("current_task_index")
    
    # 如果标记为直接结束或没有任务，直接返回
    if state.get("direct_to_end", False) or current_task_index is None or current_task_index >= len(tasks):
        logger.info("没有任务可执行或已标记直接结束，跳过工具执行")
        return state
    
    current_task = tasks[current_task_index]
    logger.info(f"执行任务: {current_task.get('description', '未知任务')}")
    
    # 直接从当前任务中获取tools_call信息
    tools_call = []
    
    if "tools_call" in current_task and current_task.get("tools_call"):
        logger.info("从当前任务的 tools_call 字段中获取工具调用信息")
        tools_call = current_task.get("tools_call", [])
        logger.info(f"获取到 {len(tools_call)} 个工具调用信息")
    
    # 如果仍然没有找到工具调用信息，则任务失败
    if not tools_call:
        logger.warning("没有找到工具调用信息，跳过工具执行")
        # 更新任务状态为失败
        tasks[current_task_index]["status"] = "failed"
        tasks[current_task_index]["result"] = "没有找到工具调用信息"
        task_manager["tasks"] = tasks
        return {
            "messages": messages,
            "task_manager": task_manager,
            "execution_status": "thinking"
        }
    
    # 记录任务的工具信息 - 使用第一个工具的名称作为主要工具名
    first_tool = tools_call[0]
    if isinstance(first_tool, dict):
        tool_name = first_tool.get("tool", "")
        tool_args = first_tool.get("tool_args", {})
    else:
        tool_name = getattr(first_tool, "name", "未知工具")
        tool_args = getattr(first_tool, "args", {})
    
    # 记录工具信息到任务 - 保留第一个工具信息兼容旧逻辑
    tasks[current_task_index]["tool_name"] = tool_name
    tasks[current_task_index]["tool_args"] = json.dumps(tool_args, ensure_ascii=False)
    
    # 准备记录所有工具执行结果
    all_tool_results = []
    
    # 依次执行每个工具
    for tool_index, tool_call in enumerate(tools_call):
        # 获取工具名称和参数
        if isinstance(tool_call, dict):
            current_tool_name = tool_call.get("tool", "")
            current_tool_args = tool_call.get("tool_args", {})
        else:
            current_tool_name = getattr(tool_call, "name", "未知工具")
            current_tool_args = getattr(tool_call, "args", {})
        
        logger.info(f"调用工具 {tool_index+1}/{len(tools_call)}: {current_tool_name}")
        logger.info(f"工具参数: {json.dumps(current_tool_args, ensure_ascii=False)}")
        
        try:
            # 直接调用工具函数而不使用AI模型
            global tools
            
            # 查找匹配的工具
            target_tool = None
            for tool in tools:
                if hasattr(tool, "name") and tool.name == current_tool_name:
                    target_tool = tool
                    break
            
            if target_tool is None:
                raise Exception(f"未找到工具: {current_tool_name}")
            
            # 尝试将工具参数从字符串转换为对象（如果是字符串）
            if isinstance(current_tool_args, str):
                try:
                    current_tool_args = json.loads(current_tool_args)
                except:
                    # 如果不是有效的 JSON，保持原样
                    pass
            
            logger.info(f"直接调用工具 {current_tool_name} 的execute方法...")
            # 直接调用工具的ainvoke方法
            # 如果不是第一个工具，工具执行参数ref字符串第二个数字加1构造成新的ref字符串
            if tool_index > 0:
                start_with_ref = current_tool_args["ref"][0]
                old_ref_num = int(current_tool_args["ref"][1])
                # 第二位以后
                end_with_ref = current_tool_args["ref"][2:] 
                new_ref_num = old_ref_num + tool_index
                current_tool_args["ref"] = f"{start_with_ref}{new_ref_num}{end_with_ref}"
            tool_result = await target_tool.ainvoke(current_tool_args)
            
            # 处理工具执行结果
            if tool_result is None:
                tool_result = f"工具 {current_tool_name} 执行完成，但没有返回结果"
            elif not isinstance(tool_result, str):
                tool_result = str(tool_result)
            
            logger.info(f"工具 {current_tool_name} 执行结果: {tool_result[:200]}...")
            
            # 添加到工具结果列表
            all_tool_results.append({
                "tool_name": current_tool_name,
                "tool_args": json.dumps(current_tool_args, ensure_ascii=False),
                "result": tool_result
            })
            
            # 直接更新任务的snapshot字段，因为MCP工具结果就是页面快照
            logger.info(f"更新任务的页面快照 (来自工具 {current_tool_name})")
            tasks[current_task_index]["snapshot"] = tool_result
            
        except Exception as e:
            logger.error(f"执行工具 {current_tool_name} 时出错: {str(e)}")
            all_tool_results.append({
                "tool_name": current_tool_name,
                "tool_args": json.dumps(current_tool_args, ensure_ascii=False),
                "result": f"执行出错: {str(e)}"
            })
            # 继续执行下一个工具，不中断整个过程
    
    # 合并所有工具执行结果到一个字符串
    combined_result = ""
    for idx, result in enumerate(all_tool_results):
        combined_result += f"\n工具{idx+1} - {result['tool_name']}:\n"
        combined_result += f"参数: {result['tool_args']}\n"
        combined_result += f"结果: {result['result']}\n"
        combined_result += "-" * 40 + "\n"
    
    # 根据任务类型执行特定的后处理
    task_type = current_task.get("task_type", "动作执行")
    
    # 处理信息检索类型任务 - 提取数据
    if task_type == "信息检索" and "snapshot" in tasks[current_task_index]:
        snapshot_result = tasks[current_task_index]["snapshot"]
        # 如果有快照，使用LLM提取数据
        extracted_data = await extract_data_from_snapshot(
            snapshot_result, current_task.get("description", "")
        )
        if extracted_data:
            tasks[current_task_index]["extracted_data"] = extracted_data
            # 将提取的数据作为最终结果的一部分
            combined_result += f"\n提取的数据:\n{extracted_data}\n"
    
    # 完成任务处理
    tasks[current_task_index]["status"] = "completed"
    tasks[current_task_index]["result"] = combined_result.strip()
    tasks[current_task_index]["last_execution"] = True  # 标记为最近执行的任务
    
    # 更新任务管理器
    task_manager["tasks"] = tasks
    
    # 返回更新后的状态
    return {
        "messages": messages,
        "task_manager": task_manager,
        "execution_status": "thinking"  # 切换到思考状态规划下一个任务
    }

# 辅助函数：获取页面快照 - 仅在需要时使用
async def get_page_snapshot():
    global tools
    # 检查是否有browser_snapshot工具
    browser_snapshot_tool = None
    for tool in tools:
        if hasattr(tool, "name") and tool.name == "browser_snapshot":
            browser_snapshot_tool = tool
            break
    
    if browser_snapshot_tool:
        try:
            return await browser_snapshot_tool.ainvoke({})
        except Exception as e:
            logger.error(f"获取页面快照时出错: {str(e)}")
            return None
    
    return None

# 辅助函数：从快照中提取数据
async def extract_data_from_snapshot(snapshot_result, task_description):
    global model
    
    # 构建提取数据的提示
    extraction_prompt = f"""
你是一个专业的网页数据提取助手。请根据当前任务需求，从页面结果中提取所需的信息。

任务描述: {task_description}

页面结果内容:
{snapshot_result[:8000] if len(str(snapshot_result)) > 8000 else snapshot_result}

这个页面结果是由MCP工具返回的完整结果，包含页面的HTML结构以及其他相关信息。请分析这个结果，提取与任务相关的关键信息。

请以清晰、结构化的方式呈现提取的数据。如果是搜索结果，请包含标题、链接和摘要。
如果是产品数据，请包含名称、价格、评分等关键信息。
如果页面中没有找到相关信息，请明确说明。

请以JSON格式返回提取的数据，或以人类易读的格式组织信息。
"""
    
    try:
        # 调用LLM提取数据
        extraction_messages = [
            HumanMessage(content=f"任务: {task_description}"),
            AIMessage(content=extraction_prompt),
        ]
        
        extraction_response = await model.ainvoke(extraction_messages)
        extraction_result = (
            extraction_response.content
            if hasattr(extraction_response, "content")
            else str(extraction_response)
        )
        
        logger.info("成功从页面结果中提取数据")
        return extraction_result
    except Exception as e:
        logger.error(f"提取数据时出错: {str(e)}")
        return None


# 思考节点 - 生成或更新任务列表 (异步版本) - 采用单任务规划策略
async def think(state: AgentState) -> AgentState:
    logger.info("=== 进入思考节点 ===")

    messages = state["messages"]
    task_manager = state.get("task_manager", {"tasks": [], "current_task_index": None})
    tasks = task_manager.get("tasks", [])
    current_task_index = task_manager.get("current_task_index")
    execution_status = state.get("execution_status", "thinking")

    logger.info(
        f"当前状态: 任务数量={len(tasks)}, 当前任务索引={current_task_index}, 执行状态={execution_status}"
    )

    # 获取工具描述 - 使用全局工具列表
    global tools

    # 确保工具列表不为空
    if not tools:
        logger.warning("工具列表为空，正在尝试重新初始化...")
        _, tools_list = await initialize_tool_node()
        tools = tools_list

    tools_description = "\n".join(
        [
            f"- {tool.name}: {tool.description}，参数schema: {tool.args_schema}"
            for tool in tools
            if hasattr(tool, "name") and hasattr(tool, "description") and hasattr(tool, "args_schema")
        ]
    )

    # 获取原始用户请求
    original_request = (
        messages[0].content
        if messages and hasattr(messages[0], "content")
        else "用户请求"
    )

    # 构建任务历史记录 - 但只用于模型的上下文信息，不会添加到消息历史
    task_history = []
    for i, task in enumerate(tasks):
        status = task.get("status", "unknown")
        tool_name = task.get("tool_name", task.get("tool", "unknown"))
        task_desc = task.get("description", "未知任务")
        task_type = task.get("task_type", "未指定")

        # 对于已完成的任务，添加简要结果
        if status == "completed":
            task_entry = f"任务{i+1}: '{task_desc}' - 类型: {task_type} - 状态: 已完成 - 工具: {tool_name}"
            task_history.append(task_entry)
        # 对于失败的任务，添加失败原因
        elif status == "failed":
            reason = task.get("result", "未知失败原因")
            task_entry = f"任务{i+1}: '{task_desc}' - 类型: {task_type} - 状态: 失败 - 原因: {reason}"
            task_history.append(task_entry)
        # 其他状态（进行中、等待中）
        else:
            task_entry = (
                f"任务{i+1}: '{task_desc}' - 类型: {task_type} - 状态: {status}"
            )
            task_history.append(task_entry)

    task_history_text = "\n".join(task_history) if task_history else "尚未执行任何任务"

    # 获取最近执行的任务结果 - 特别强调最近的工具执行结果
    last_completed_task = None
    last_completed_index = -1

    # 查找策略1: 首先查找带有last_execution标记的任务
    logger.info("尝试查找带有last_execution标记的任务")
    for i, task in enumerate(tasks):
        if task.get("last_execution", False):
            last_completed_task = task
            last_completed_index = i  # 正确使用枚举索引 i
            logger.info(f"找到带last_execution标记的任务: {task.get('description', '未知任务')}")
            break
    
    # 查找策略2: 如果没找到带标记的任务，则使用current_task_index
    if last_completed_task is None and current_task_index is not None and current_task_index < len(tasks):
        last_completed_task = tasks[current_task_index]
        last_completed_index = current_task_index  # 设置索引为当前任务索引
        logger.info(f"使用当前任务索引指向的任务: {last_completed_task.get('description', '未知任务')}")
    
    # 查找策略3: 如果仍未找到或当前任务不是已完成状态，尝试查找最近完成的任务
    if last_completed_task is None or last_completed_task.get("status") != "completed":
        logger.info("尝试查找最近完成的任务")
        completed_tasks = [t for t in tasks if t.get("status") == "completed"]
        if completed_tasks:
            last_completed_task = completed_tasks[-1]  # 最后完成的任务
            # 找到任务在列表中的索引
            for i, task in enumerate(tasks):
                if task is last_completed_task:
                    last_completed_index = i
                    break
            logger.info(f"找到最近完成的任务: {last_completed_task.get('description', '未知任务')}")
    
    # 特别突出最近一次工具执行的结果，根据任务类型设置不同的内容
    recent_task_result = ""
    if last_completed_task:
        tool_name = last_completed_task.get(
            "tool_name", last_completed_task.get("tool", "unknown")
        )
        tool_args = last_completed_task.get("tool_args", "")
        result = last_completed_task.get("result", "无结果")
        task_type = last_completed_task.get("task_type", "未指定")
        task_desc = last_completed_task.get("description", "未知任务")

        # 根据任务类型设置不同的内容
        if task_type == "动作执行":
            # 对于动作执行任务，创建页面状态描述，而不是直接显示原始快照
            snapshot = last_completed_task.get("snapshot", "")
            
            recent_task_result = f"""
当前页面状态: {snapshot}
"""
        elif task_type == "信息检索":
            # 对于信息检索任务，展示检索结果
            extracted_data = last_completed_task.get("extracted_data", result)
            recent_task_result = f"""
当前任务检索结果为: {extracted_data}
"""
        else:
            # 对于未指定类型的任务，使用常规格式
            # 获取snapshot，以避免引用未定义的变量
            snapshot = last_completed_task.get("snapshot", "")
            
            recent_task_result = f"""
最近执行的任务 (任务{last_completed_index+1}): '{task_desc}'
任务类型: {task_type}
使用工具: {tool_name}
工具参数: {tool_args}
当前页面状态: {snapshot}
"""

    # 构建提示模板，根据是否有已完成任务进行调整
    if not task_history:
        # 初始规划 - 第一个任务
        system_prompt = f"""
你是一个智能浏览器自动化助手，负责规划和执行连续性的浏览器操作任务。
根据用户请求，请规划下一步应该执行的单个任务。

用户的原始请求是: "{original_request}"

你只能使用以下工具来执行任务：
{tools_description}

请分析用户请求，确定接下来需要采取的最合适的单个浏览器操作。你不应该规划完整的任务列表，只需规划当前最应该执行的一个任务。

每个任务必须归类为以下类型之一：
1. "动作执行": 如打开网页、点击按钮、输入文本等操作性任务
2. "信息检索": 如采集搜索结果、提取网页数据等获取信息的任务

任务合并与拆分准则:
- 合并情况: 如表单填写（多个输入框）、连续点击等无需页面刷新的操作，可以规划为一个任务
  例如: "在登录表单中输入用户名xxx和密码yyy，然后点击登录按钮"
- 拆分情况: 涉及页面跳转、等待新内容加载、条件判断的操作，应该拆分为多个任务
  例如: "打开网页"和"检索网页内容"应当拆分为两个独立任务

新的任务格式规范：
1. 任务必须包含以下字段:
   - "task": 具体任务内容的描述
   - "task_type": 任务类型，必须是"动作执行"或"信息检索"中的一个
   - "tools_call": 工具调用数组，包含要执行的工具调用列表
2. 其中 tools_call 是一个数组，每个元素包含:
   - "tool": 推荐使用的工具名称
   - "task_action": 该工具的具体动作描述
   - "tool_args": 该工具的参数，参数的schema请参考工具的定义

任务内容应该直接用动词开头，例如"打开页面..."、"点击按钮..."、"采集数据..."
只生成能够用以上工具直接执行的任务，并且任务应该具体明确，包含足够的信息让工具能理解。

返回格式必须是：
```json
{{
  "task": "具体任务内容",
  "task_type": "动作执行或信息检索",
  "tools_call": [
    {{
      "tool": "推荐的工具名称",
      "task_action": "具体工具动作描述",
      "tool_args": {{"参数1": "参数1的值", "参数2": "参数2的值"}}
    }}
  ]
}}
```

请直接返回JSON对象，不要添加额外解释。
"""
    else:
        # 后续规划 - 基于已有的任务结果，特别是最近的执行结果
        system_prompt = f"""
你是一个智能浏览器自动化助手，负责规划和执行连续性的浏览器操作任务。
根据用户请求和已完成任务的结果，特别是最近执行的任务结果，请规划下一步应该执行的单个任务。

用户的原始请求是: "{original_request}"

历史任务执行情况:
{task_history_text}

{recent_task_result}

你只能使用以下工具来执行任务：
{tools_description}

请分析用户请求和已完成任务的结果，特别是最近执行的任务结果，确定接下来需要采取的最合适的单个浏览器操作。你不应该规划完整的任务列表，只需规划当前最应该执行的一个任务。

每个任务必须归类为以下类型之一：
1. "动作执行": 如打开网页、点击按钮、输入文本等操作性任务
2. "信息检索": 如采集搜索结果、提取网页数据等获取信息的任务
3. "任务检查": 如检查页面是否已经加载完成，是否需要获取页面快照等，特别是在新打开页面这种情况下

任务合并与拆分准则:
- 已知在任务执行阶段，系统支持在一个任务中执行多个连续的工具操作
- 合并情况: 如表单填写（多个输入框）、连续点击等无需页面刷新的操作，可以规划为一个任务
  例如: "在登录表单中输入用户名xxx和密码yyy，然后点击登录按钮"
- 拆分情况: 涉及页面跳转、等待新内容加载、条件判断的操作，应该拆分为多个任务
  例如: "打开网页"和"检索网页内容"应当拆分为两个独立任务

特别注意：
1. 浏览器操作是连续性的，每一步应该基于上一步的结果
2. 任务执行阶段已支持在一个任务中执行多个连续工具操作，无需为简单连续操作拆分多个任务
3. 最近完成的任务返回的结果是你确定下一步行动的最重要依据
4. 如果上一步操作失败或返回了错误信息，可能需要调整策略或尝试不同的方法
5. 如果需要分析页面内容，应明确规划一个"信息检索"类型的任务
6. 检索某个页面内容时需检测当前tab与检索的页面是否一致，如果一致则直接检索，否则需要先切换到检索的页面
7. 每次MCP工具执行后会自动返回页面快照，不需要专门规划一个获取页面状态的任务或者工具执行动作
8. 非必要情况不要规划获取页面快照的任务
9. 工具如果返回- Page Snapshot
```yaml
null
这种格式代表页面还没加载完成吗，无法获取页面快照
10. 如果无法检测到页面快照真实内容（排除页面title，url），则需要规划一个获取页面快照的任务
11. 需要耗时较长的任务，请规划为"任务检查"类型，通过工具获取页面快照，如果页面快照中没有检测到页面加载完成，则需要规划一个获取页面快照的任务

新的任务格式规范：
1. 任务必须包含以下字段:
   - "task": 具体任务内容的描述
   - "task_type": 任务类型，必须是"动作执行"、"信息检索"、"任务检查"中的一个
   - "tools_call": 工具调用数组，包含要执行的工具调用列表
2. 其中 tools_call 是一个数组，每个元素包含:
   - "tool": 推荐使用的工具名称
   - "task_action": 该工具的具体动作描述  
   - "tool_args": 该工具的参数
3. 执行动作时，工具参数必须从页面快照中提取，不准随意编造
4. 工具参数的schema请参考工具的定义，请不要随意编造参数以及遗漏参数，严格按照schem的格式来，包括key的名称，不要合并key

任务内容应该直接用动词开头，例如"打开页面..."、"点击按钮..."、"采集数据..."
任务应该基于已完成任务的结果，特别是最近执行的结果。
如果已有结果足够回答用户问题，请返回空对象 {{}}。

返回格式必须是：
```json
{{
  "task": "具体任务内容",
  "task_type": "动作执行或信息检索或任务检查",
  "tools_call": [
    {{
      "tool": "推荐使用的工具名称",
      "task_action": "具体工具动作描述",
      "tool_args": {{"参数1": "参数1的值", "参数2": "参数2的值"}}
    }}
  ]
}}
```

或者如果不需要更多任务:
```json
{{}}
```

请直接返回JSON对象，不要添加额外解释。
"""

        logger.info(f"规划提示: {system_prompt}")

    # 准备消息上下文 - 确保包含原始请求
    # 优化：只使用原始用户请求和最近的工具执行结果，避免上下文过大
    thinking_messages = [HumanMessage(content=original_request)]

    # 直接添加最近的工具执行结果作为上下文，而不是所有历史结果
    if recent_task_result:
        thinking_messages.append(
            AIMessage(content=f"最近的浏览器状态:\n{recent_task_result}")
        )

    # 添加任务历史概要信息，但不包含详细结果
    if task_history:
        thinking_messages.append(
            AIMessage(content=f"已执行任务概要:\n{task_history_text}")
        )

    # 添加系统提示
    thinking_messages.append(AIMessage(content=system_prompt))

    # 调用LLM规划下一个任务
    logger.info("调用LLM规划下一个任务...")
    response = await model.ainvoke(thinking_messages)
    logger.info(f"LLM响应: {response.content[:200]}...，消耗token: {response.response_metadata.get('token_usage', {}).get('total_tokens', 0)}")

    # 解析响应获取任务
    task_json = response.content

    # 提取JSON部分
    import re

    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", task_json)
    if json_match:
        task_json = json_match.group(1)
    else:
        # 尝试直接解析整个响应
        task_json = task_json.strip()

    # 解析JSON
    try:
        import json

        task_data = json.loads(task_json)
        logger.info(f"成功解析任务JSON: {task_data}")

        # 检查是否有任务要执行
        if not task_data or "task" not in task_data:
            logger.info("没有更多任务需要执行，准备生成最终总结")
            # 如果没有任务，生成最终总结
            # 关键修改：直接返回generate_final_summary的结果，不做额外处理
            final_summary_state = await generate_final_summary(messages, tasks, original_request)
            logger.info("直接返回最终总结状态")
            return final_summary_state

        # 创建新任务
        task_description = task_data.get("task", "")
        task_type = task_data.get("task_type", "动作执行")  # 默认为动作执行

        # 验证任务类型
        if task_type not in ["动作执行", "信息检索", "任务检查"]:
            logger.warning(f"任务类型 '{task_type}' 无效，设置为默认值 '动作执行'")
            task_type = "动作执行"

        # 获取工具调用信息
        tools_call = task_data.get("tools_call", [])
        
        # 创建格式化描述，但不再将工具信息添加到描述中
        formatted_description = task_description

        # 创建新的任务对象，包含任务类型和工具调用数组
        new_task = {
            "description": formatted_description,
            "task_type": task_type,
            "status": "pending",
            "tools_call": tools_call  # 存储整个工具调用数组
        }

        logger.info(f"规划了新任务: {new_task['description']} - 类型: {task_type} - 工具调用数量: {len(tools_call)}")

        # 添加新任务到任务列表
        tasks.append(new_task)
        
        # 此时清除任务中的last_execution标记，确保只有新执行的任务会有这个标记
        for task in tasks:
            if "last_execution" in task:
                del task["last_execution"]
        
        # 更新任务管理器
        task_manager["tasks"] = tasks
        task_manager["current_task_index"] = len(tasks) - 1  # 指向新添加的任务

        # 设置执行状态
        return {
            "messages": messages,
            "task_manager": task_manager,
            "execution_status": "executing",
        }

    except Exception as e:
        logger.error(f"解析任务JSON失败: {str(e)}")

        # 如果解析失败但已经完成了一些任务，尝试生成最终总结
        if task_history:
            logger.info("任务解析失败，但已有完成的任务，准备生成最终总结")
            # 同样直接返回最终总结状态
            final_summary_state = await generate_final_summary(messages, tasks, original_request)
            logger.info("任务解析失败，直接返回最终总结状态")
            return final_summary_state

        # 初始任务也失败的情况，创建一个通用的搜索任务作为后备
        fallback_task = {
            "description": f"搜索有关 '{original_request}' 的信息",
            "task_type": "信息检索",  # 设置为信息检索类型
            "status": "pending",
            "tools_call": [  # 添加默认的工具调用数组
                {
                    "tool": "search_web",
                    "task_action": f"搜索有关 '{original_request}' 的信息",
                    "tool_args": original_request
                }
            ]
        }

        logger.info(f"解析失败，使用后备任务: {fallback_task['description']}")
        tasks.append(fallback_task)
        
        # 清除任务中的last_execution标记
        for task in tasks:
            if "last_execution" in task:
                del task["last_execution"]
        
        # 更新任务管理器
        task_manager["tasks"] = tasks
        task_manager["current_task_index"] = len(tasks) - 1

        return {
            "messages": messages,
            "task_manager": task_manager,
            "execution_status": "executing",
        }


# 生成最终总结的辅助函数
async def generate_final_summary(messages, tasks, original_request):
    # 收集所有任务的结果，按执行顺序排列
    completed_tasks = [task for task in tasks if task.get("status") == "completed"]
    failed_tasks = [task for task in tasks if task.get("status") == "failed"]

    # 构建任务执行历史的详细摘要
    task_details = []
    information_results = []  # 存储信息检索任务的结果

    for i, task in enumerate(tasks):
        status = task.get("status", "未知")
        tool_name = task.get("tool_name", task.get("tool", "未知工具"))
        description = task.get("description", "未知任务")
        task_type = task.get("task_type", "未指定")

        if status == "completed":
            tool_args = task.get("tool_args", "")
            result = task.get("result", "无结果")

            # 根据任务类型添加特定信息
            if task_type == "动作执行":
                has_snapshot = "snapshot" in task
                
                # 如果有快照，提取页面标题
                page_info = ""
                if has_snapshot:
                    snapshot = task.get("snapshot", "")
                    if snapshot and len(snapshot) > 500:
                        title_match = re.search(r"<title>(.*?)</title>", snapshot, re.IGNORECASE | re.DOTALL)
                        page_title = title_match.group(1) if title_match else "未知页面"
                        page_info = f"（当前页面: {page_title}）"
                    else:
                        page_info = "（页面已加载）"
                
                task_entry = f"""
任务 {i+1}: {description}
状态: 已完成
类型: {task_type} {page_info}
工具: {tool_name}
参数: {tool_args}
执行结果: {result}
"""
            elif task_type == "信息检索":
                extracted_data = task.get("extracted_data", "")
                if extracted_data:
                    task_entry = f"""
任务 {i+1}: {description}
状态: 已完成
类型: {task_type}
工具: {tool_name}
参数: {tool_args}
提取的数据: {extracted_data}
"""
                    # 保存信息检索结果用于总结
                    information_results.append(
                        {"task": description, "data": extracted_data}
                    )
                else:
                    task_entry = f"""
任务 {i+1}: {description}
状态: 已完成
类型: {task_type}
工具: {tool_name}
参数: {tool_args}
执行结果: {result}
"""
            else:
                task_entry = f"""
任务 {i+1}: {description}
状态: 已完成
类型: {task_type}
工具: {tool_name}
参数: {tool_args}
执行结果: {result}
"""
            task_details.append(task_entry)
        elif status == "failed":
            reason = task.get("result", "未知失败原因")
            task_entry = f"""
任务 {i+1}: {description}
状态: 失败
类型: {task_type}
原因: {reason}
"""
            task_details.append(task_entry)

    task_history = "\n".join(task_details)

    # 将信息检索结果格式化为总结的输入
    information_summary = ""
    if information_results:
        information_entries = []
        for i, info in enumerate(information_results):
            information_entries.append(
                f"""
信息 {i+1}:
任务: {info['task']}
数据: {info['data']}
"""
            )
        information_summary = "\n".join(information_entries)

    # 构建提示，生成人性化的回复
    summary_prompt = f"""
用户的原始请求是: "{original_request}"

我已经执行了以下浏览器自动化任务来回答用户的请求:

{task_history}

{"提取的关键信息:" if information_results else ""}
{information_summary if information_results else ""}

请根据这些任务执行的历史和结果，特别是提取的信息，生成一个简洁、友好且人性化的回复。

你的回复应该：
1. 直接回答用户的问题
2. 简要总结浏览器自动化执行的关键步骤和发现
3. 重点突出从网页提取的关键信息
4. 如果有任何失败的任务，解释为什么会失败以及可能的解决方案
5. 不要用"我执行了哪些任务"这样的表述，而是直接给出结果和解释
6. 整合所有信息，给出一个连贯、有价值的回答
7. 如果浏览器操作没有找到预期的信息，请坦率说明

如果任务是网页浏览相关的，你可以对内容做一个简洁的总结，并提取最重要的信息以回答用户的问题。
"""

    # 调用LLM生成人性化回复 (异步)
    logger.info("调用LLM生成最终人性化回复...")

    # 优化：创建干净的消息列表用于总结生成，避免上下文污染
    # 只使用原始用户请求和任务历史信息
    if messages and len(messages) > 0:
        original_user_message = messages[0]  # 保留原始用户请求
        summary_message_list = [original_user_message]
    else:
        summary_message_list = [HumanMessage(content=original_request)]

    # 添加任务历史作为上下文
    summary_message_list.append(AIMessage(content=f"任务执行历史:\n{task_history}"))

    # 添加信息检索结果作为上下文（如果有）
    if information_results:
        summary_message_list.append(
            AIMessage(content=f"提取的信息:\n{information_summary}")
        )

    # 添加总结提示
    summary_message_list.append(AIMessage(content=summary_prompt))

    try:
        summary_response = await model.ainvoke(summary_message_list)
        summary_content = (
            summary_response.content
            if hasattr(summary_response, "content")
            else str(summary_response)
        )
        logger.info(f"生成的人性化回复: {summary_content[:100]}...")
    except Exception as e:
        logger.error(f"生成人性化回复时出错: {str(e)}")
        summary_content = f"已完成所有可执行的任务，任务结果如下:\n{task_history}"
        if information_results:
            summary_content += f"\n\n提取的信息:\n{information_summary}"

    summary_message = AIMessage(content=summary_content)

    # 返回干净的消息历史 - 只保留原始用户请求和最终总结
    if messages and len(messages) > 0:
        final_messages = [messages[0], summary_message]
    else:
        final_messages = [HumanMessage(content=original_request), summary_message]

    # 创建task_manager对象
    task_manager = {
        "tasks": tasks,
        "current_task_index": None
    }
    
    # 关键修改：直接在这里将执行状态设置为"finished"，并显式标记避免进入action节点
    logger.info("生成最终总结完成，直接标记工作流执行完成")
    return {
        "messages": final_messages,  # 使用优化的消息历史
        "task_manager": task_manager,
        "execution_status": "finished",  # 明确设置为finished状态
        "direct_to_end": True,  # 添加标记，表示应该直接结束工作流
    }


# 决策函数 - 根据动作结果决定下一步操作
async def decide_next_step(state: AgentState) -> Literal["think", "tool", "end"]:
    """根据动作结果决定下一步操作"""
    # 检查是否有直接结束标记
    if state.get("direct_to_end", False):
        logger.info("检测到direct_to_end标记，决定结束工作流")
        return "end"
        
    # 提取必要的信息
    task_manager = state.get("task_manager")
    execution_status = state.get("execution_status", "thinking")
    
    # 检查是否存在任务管理器
    if not task_manager:
        logger.error("任务管理器不存在，无法决定下一步操作")
        state["direct_to_end"] = True
        return "end"
    
    # 检查执行状态
    if execution_status == "finished":
        logger.info("执行状态已完成，决定结束工作流")
        return "end"
    
    # 检查当前任务
    tasks = task_manager.get("tasks", [])
    current_task_index = task_manager.get("current_task_index")
    
    # 如果当前任务索引无效，返回思考节点
    if current_task_index is None or current_task_index >= len(tasks):
        logger.info("没有当前任务，决定进入思考节点")
        return "think"
    
    # 获取当前任务
    current_task = tasks[current_task_index]
    
    # 检查任务状态
    task_status = current_task.get("status", "unknown")
    
    if task_status == "pending" or task_status == "in_progress":
        # 任务待执行或正在执行中，进入工具执行节点
        logger.info(f"当前任务'{current_task.get('description', '未知任务')}'状态为{task_status}，进入工具执行节点")
        return "tool"
    elif task_status == "completed":
        # 任务已完成，检查是否有下一个任务
        if current_task_index < len(tasks) - 1:
            # 移动到下一个任务
            task_manager["current_task_index"] = current_task_index + 1
            logger.info("当前任务已完成，移动到下一个任务")
            return "think"
        else:
            # 所有任务已完成，结束工作流
            logger.info("所有任务已完成，决定结束工作流")
            state["execution_status"] = "finished"
            return "end"
    elif task_status == "failed":
        # 任务失败，尝试重新规划
        logger.warning(f"任务'{current_task.get('description', '未知任务')}'失败，重新进入思考节点")
        return "think"
    else:
        # 未知状态，默认进入思考节点
        logger.warning(f"未知任务状态: {task_status}，默认进入思考节点")
        return "think"


# 决策函数 - 用于after_tool节点的后续处理决策 (异步版本)
async def after_tool_decision(state: AgentState) -> Literal["think", "end"]:
    execution_status = state.get("execution_status", "thinking")
    
    # 检查是否标记为直接结束
    if state.get("direct_to_end", False):
        logger.info("检测到direct_to_end标记，直接结束工作流")
        return "end"

    # 如果执行状态已完成，直接结束流程
    if execution_status == "finished":
        logger.info("执行状态已完成，决定结束流程")
        return "end"
    else:
        logger.info("执行状态为思考或执行中，返回思考节点")
        return "think"


# 报告节点 - 统计和汇总任务信息
async def report(state: AgentState) -> AgentState:
    logger.info("=== 进入Report节点 ===")
    
    messages = state["messages"]
    task_manager = state.get("task_manager", {"tasks": [], "current_task_index": None})
    tasks = task_manager.get("tasks", [])
    current_task_index = task_manager.get("current_task_index")
    
    logger.info(f"当前状态: 任务数量={len(tasks)}, 当前任务索引={current_task_index}")
    
    # 如果有direct_to_end标记，直接返回状态
    if state.get("direct_to_end", False):
        logger.info("检测到direct_to_end标记，Report节点直接传递状态")
        return state
    
    # 统计所有任务的状态
    completed_tasks = [task for task in tasks if task.get("status") == "completed"]
    failed_tasks = [task for task in tasks if task.get("status") == "failed"]
    pending_tasks = [task for task in tasks if task.get("status") == "pending"]
    in_progress_tasks = [task for task in tasks if task.get("status") == "in_progress"]
    
    # 打印任务统计信息
    logger.info(f"任务总数: {len(tasks)}")
    logger.info(f"已完成任务: {len(completed_tasks)}")
    logger.info(f"失败任务: {len(failed_tasks)}")
    logger.info(f"待执行任务: {len(pending_tasks)}")
    logger.info(f"执行中任务: {len(in_progress_tasks)}")
    
    # 获取最近执行的任务详情
    recent_task = None
    
    # 查找带有last_execution标记的任务
    for task in tasks:
        if task.get("last_execution", False):
            recent_task = task
            logger.info(f"找到带last_execution标记的任务: {task.get('description', '未知任务')}")
            break
    
    # 如果没找到带标记的任务，则使用current_task_index
    if recent_task is None and current_task_index is not None and current_task_index < len(tasks):
        recent_task = tasks[current_task_index]
        logger.info(f"使用当前任务索引指向的任务: {recent_task.get('description', '未知任务')}")
    
    # 如果仍未找到或当前任务不是已完成状态，尝试查找最近完成的任务
    if recent_task is None or recent_task.get("status") != "completed":
        logger.info("尝试查找最近完成的任务")
        completed_tasks = [t for t in tasks if t.get("status") == "completed"]
        if completed_tasks:
            recent_task = completed_tasks[-1]  # 最后完成的任务
            logger.info(f"找到最近完成的任务: {recent_task.get('description', '未知任务')}")
    
    # 打印最近任务的详细信息
    if recent_task:
        task_status = recent_task.get("status", "未知")
        task_desc = recent_task.get("description", "未知任务")
        task_type = recent_task.get("task_type", "未指定")
        tool_name = recent_task.get("tool_name", recent_task.get("tool", "未知工具"))
        
        logger.info(f"最近执行任务: {task_desc}")
        logger.info(f"任务状态: {task_status}")
        logger.info(f"任务类型: {task_type}")
        logger.info(f"使用工具: {tool_name}")
        
        # 根据任务类型输出更多详情
        if task_status == "completed":
            result = recent_task.get("result", "")
            if task_type == "信息检索":
                extracted_data = recent_task.get("extracted_data", "")
                if extracted_data:
                    logger.info(f"提取的数据: {extracted_data[:150]}...")
            else:
                # 对于动作执行任务，处理快照信息
                if "snapshot" in recent_task:
                    snapshot = recent_task.get("snapshot", "")
                    # 提取页面标题而不是显示完整HTML
                    if snapshot and len(snapshot) > 500:
                        title_match = re.search(r"<title>(.*?)</title>", snapshot, re.IGNORECASE | re.DOTALL)
                        page_title = title_match.group(1) if title_match else "未知页面"
                        logger.info(f"页面标题: {page_title}")
                    else:
                        logger.info("页面已加载")
                
                if len(result) > 150:
                    logger.info(f"执行结果: {result[:150]}...")
                else:
                    logger.info(f"执行结果: {result}")
        elif task_status == "failed":
            logger.info(f"失败原因: {recent_task.get('result', '未知')}")
    
    # 准备下一步执行状态
    return {
        "messages": messages,
        "task_manager": task_manager,
        "execution_status": "thinking"  # 设置为思考状态
    }


# 修改构建工作流图函数
def build_agent_workflow():
    logger.info("开始构建工作流图")
    # 创建状态图 - 使用异步模式
    from langgraph.graph import StateGraph

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("think", think)
    workflow.add_node("action", action)
    workflow.add_node("tool_execution", tool_execution)  # 合并后的工具执行节点
    workflow.add_node("report", report)  # 报告节点
    logger.info("已添加节点: think, action, tool_execution, report")

    # 添加入口边 - 从START到think
    workflow.add_edge(START, "think")
    logger.info("已添加入口边: START -> think")

    # 添加基本边
    workflow.add_edge("think", "action")
    logger.info("已添加边: think -> action")

    # 添加条件边 - 从action节点到其他节点
    workflow.add_conditional_edges(
        "action", decide_next_step, {
            "think": "think", 
            "tool": "tool_execution",  # 重定向到合并的工具执行节点
            "end": END
        }
    )
    logger.info("已添加条件边: action -> [think, tool_execution, END]")
    
    # 添加条件边 - 从tool_execution节点到其他节点
    workflow.add_conditional_edges(
        "tool_execution", tool_execution_decision, {
            "report": "report",
            "end": END
        }
    )
    logger.info("已添加条件边: tool_execution -> [report, END]")

    # 添加条件边 - 从report节点到其他节点
    workflow.add_conditional_edges(
        "report", report_decision, {
            "think": "think",
            "end": END
        }
    )
    logger.info("已添加条件边: report -> [think, END]")

    try:
        # 编译工作流
        logger.info("编译工作流")
        compiled = workflow.compile()
        logger.info("工作流编译成功")
        return compiled
    except Exception as e:
        logger.error(f"工作流编译失败: {str(e)}")
        raise

# 工具执行决策函数
async def tool_execution_decision(state: AgentState) -> Literal["report", "end"]:
    """决定工具执行后的下一步操作"""
    # 检查是否有直接结束标记
    if state.get("direct_to_end", False):
        logger.info("检测到direct_to_end标记，决定结束工作流")
        return "end"
    
    # 默认进入报告节点
    logger.info("工具执行完成，进入报告节点")
    return "report"

# 报告决策函数
async def report_decision(state: AgentState) -> Literal["think", "end"]:
    """决定报告后的下一步操作"""
    # 检查是否有直接结束标记
    if state.get("direct_to_end", False):
        logger.info("检测到direct_to_end标记，决定结束工作流")
        return "end"
    
    # 获取执行状态
    execution_status = state.get("execution_status", "")
    if execution_status == "finished":
        logger.info("执行状态已完成，决定结束工作流")
        return "end"
    
    # 默认返回思考节点
    logger.info("报告完成，返回思考节点")
    return "think"


# 使用示例 (异步版本)
async def run_agent(user_input: str):
    logger.info(f"===== 开始执行工作流 =====")
    logger.info(f"用户输入: {user_input}")

    # 确保工作流和工具已初始化
    global agent_workflow
    if agent_workflow is None:
        logger.info("工作流尚未初始化，正在初始化...")
        agent_workflow = await setup_agent()

    # 正确初始化task_manager结构
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "task_manager": {
            "tasks": [],
            "current_task_index": None
        },
        "execution_status": "thinking",
    }

    # 执行工作流并返回最终状态 (异步)
    logger.info("开始执行工作流")
    final_state = await agent_workflow.ainvoke(initial_state)

    # 记录最终状态
    task_manager = final_state.get("task_manager", {"tasks": [], "current_task_index": None})
    tasks = task_manager.get("tasks", [])
    logger.info(f"工作流执行完成，生成了 {len(tasks)} 个任务")
    for i, task in enumerate(tasks):
        tool_info = (
            f", 工具: {task.get('tool_name', task.get('tool', 'unknown'))}"
            if "tool" in task or "tool_name" in task
            else ""
        )
        logger.info(
            f"  任务 {i+1}: {task['description']} - 状态: {task['status']}{tool_info}"
        )
        if task.get("result"):
            logger.info(f"    结果: {task.get('result')[:100]}...")

    # 记录最终的人性化回复
    messages = final_state.get("messages", [])
    if messages:
        last_message = messages[-1]
        logger.info(f"最终消息: {last_message}")
        if hasattr(last_message, "content"):
            logger.info(f"最终人性化回复: {last_message.content[:200]}...")

    logger.info(f"===== 工作流执行完成 =====")
    return final_state


# 初始化工作流为None，等待run_agent时初始化
agent_workflow = None


# 添加新的action节点函数 - 兼容单任务规划策略
async def action(state: AgentState) -> Dict:
    logger.info("=== 进入Action节点 ===")

    messages = state["messages"]
    task_manager = state.get("task_manager", {"tasks": [], "current_task_index": None})
    tasks = task_manager.get("tasks", [])
    current_task_index = task_manager.get("current_task_index")

    logger.info(f"当前状态: 任务数量={len(tasks)}, 当前任务索引={current_task_index}")

    if current_task_index is None or current_task_index >= len(tasks):
        # 没有任务可执行
        logger.info("没有任务可执行，标记为完成")
        return {
            "messages": messages,
            "task_manager": task_manager,
            "execution_status": "finished",
        }

    current_task = tasks[current_task_index]
    logger.info(f"当前任务: {current_task['description']}")

    # 将当前任务标记为进行中
    tasks[current_task_index]["status"] = "in_progress"
    logger.info(f"已将任务 {current_task_index} 标记为进行中")
    
    # 更新task_manager
    task_manager["tasks"] = tasks

    # 获取工具调用数组
    tools_call = current_task.get("tools_call", [])
    
    if not tools_call:
        logger.warning("当前任务没有工具调用信息，任务将被标记为失败")
        tasks[current_task_index]["status"] = "failed"
        tasks[current_task_index]["result"] = "任务缺少工具调用信息"
        task_manager["tasks"] = tasks
        return {
            "messages": messages,
            "task_manager": task_manager,
            "execution_status": "thinking",  # 返回思考状态以处理下一个任务
        }
    
    logger.info(f"准备执行 {len(tools_call)} 个工具调用")
    
    # 直接设置tools_call到任务中，无需转换格式
    task_manager["tasks"] = tasks
    
    # 返回更新后的状态
    return {
        "messages": messages,  # 保持消息不变
        "task_manager": task_manager,
        "execution_status": "executing",
    }


# 思考决策函数
async def think_decision(state: AgentState) -> Literal["think", "action", "tool_execution", "end"]:
    """决定思考后的下一步操作"""
    # 检查是否有直接结束标记
    if state.get("direct_to_end", False):
        logger.info("检测到direct_to_end标记，决定结束工作流")
        return "end"
    
    # 获取任务管理器
    task_manager = state.get("task_manager")
    if not task_manager:
        logger.error("任务管理器不存在，无法决定思考后的下一步操作")
        state["direct_to_end"] = True
        return "end"
    
    # 获取执行状态
    execution_status = state.get("execution_status", "")
    if execution_status == "finished" or execution_status == "failed":
        logger.info(f"执行状态为{execution_status}，决定结束工作流")
        return "end"
    
    # 获取当前任务
    tasks = task_manager.get("tasks", [])
    current_task_index = task_manager.get("current_task_index")
    
    # 如果当前任务索引无效
    if current_task_index is None or current_task_index >= len(tasks):
        if len(tasks) > 0:
            # 尝试移动到第一个任务
            task_manager["current_task_index"] = 0
            logger.info("没有当前任务，移动到第一个任务，继续思考")
            return "think"
        else:
            logger.info("没有待执行的任务，决定结束工作流")
            state["execution_status"] = "finished"
            return "end"
    
    # 获取当前任务
    current_task = tasks[current_task_index]
    
    # 如果当前任务已完成或失败，处理下一个任务
    if current_task.get("status") in ["completed", "failed"]:
        if current_task.get("status") == "failed" and state.get("retry_on_failure", False) and current_task.get("retries", 0) < state.get("max_retries", 3):
            current_task["retries"] = current_task.get("retries", 0) + 1
            current_task["status"] = "pending"
            # 更新任务列表
            tasks[current_task_index] = current_task
            task_manager["tasks"] = tasks
            logger.info(f"当前任务失败，尝试重试，重试次数：{current_task.get('retries', 0)}")
            return "think"
        elif current_task_index < len(tasks) - 1:
            # 移动到下一个任务
            task_manager["current_task_index"] = current_task_index + 1
            logger.info(f"当前任务{current_task.get('status')}，移动到下一个任务，继续思考")
            return "think"
        else:
            logger.info(f"所有任务处理完毕（最后任务状态：{current_task.get('status')}），决定结束工作流")
            state["execution_status"] = "finished" if current_task.get("status") == "completed" else "failed"
            return "end"
    
    # 获取当前思考内容
    thought = state.get("thought", "")
    
    # 检查是否需要执行工具
    tool_call = state.get("selected_tool")
    if tool_call:
        logger.info(f"检测到选择了工具：{tool_call.get('name', 'unknown')}，进入工具执行节点")
        return "tool_execution"
    
    # 检查是否有行动计划
    action_plan = state.get("action_plan", "")
    if action_plan:
        logger.info("检测到行动计划，进入行动节点")
        return "action"
    
    # 检查思考迭代次数
    think_iterations = state.get("think_iterations", 0)
    max_think_iterations = state.get("max_think_iterations", 5)
    
    if think_iterations >= max_think_iterations:
        logger.warning(f"思考迭代次数（{think_iterations}）达到上限（{max_think_iterations}），强制进入行动节点")
        state["action_plan"] = "由于思考迭代次数达到上限，需要直接采取行动或结束任务"
        return "action"
    
    # 增加思考迭代次数
    state["think_iterations"] = think_iterations + 1
    logger.info(f"继续思考，当前迭代次数：{think_iterations + 1}")
    
    # 默认继续思考
    return "think"


if __name__ == "__main__":
    import asyncio

    async def main():
        user_input = input("请输入测试指令: ")
        if not user_input:
            user_input = "计算7加8，然后再乘以2，最后除以3，并返回结果"
            print(f"使用默认指令: {user_input}")

        print("\n开始执行工作流...")
        result = await run_agent(user_input)

        print("\n执行结果汇总:")
        print("-" * 50)

        # 打印任务列表和详细结果
        tasks = result.get("tasks", [])
        if tasks:
            print("\n完整任务列表及结果:")
            for i, task in enumerate(tasks):
                status = (
                    "✓"
                    if task["status"] == "completed"
                    else "×" if task["status"] == "failed" else "..."
                )
                tool_info = (
                    f" [工具: {task.get('tool_name', task.get('tool', 'unknown'))}]"
                    if "tool" in task or "tool_name" in task
                    else ""
                )
                print(f"{i+1}. [{status}] {task['description']}{tool_info}")

                if task.get("tool_args"):
                    print(f"   参数: {task.get('tool_args')}")

                if task.get("result"):
                    # 打印完整结果，不截断
                    print(f"   结果: {task.get('result')}")

                # 分隔线
                print("-" * 30)

        # 打印最终消息
        messages = result.get("messages", [])
        if messages:
            print("\n最终回应:")
            last_message = messages[-1] if messages else None
            if last_message and hasattr(last_message, "content"):
                content = last_message.content
                # 打印完整响应，不截断
                print(f"回应: {content}")

        print("-" * 50)
        print("工作流执行完成!")

    # 运行异步主函数
    asyncio.run(main())
