# 智能浏览器自动化助手

一个基于LangGraph的智能浏览器自动化系统，通过大语言模型规划和执行浏览器操作，解决用户复杂任务。

## 项目结构

- `main.py`: 主入口文件，处理用户输入并运行agent
- `graph.py`: 核心工作流定义，包含工作流节点和状态转换逻辑

## 执行逻辑流程

```mermaid
graph TD
    A[开始] --> B[输入用户指令]
    B --> C[初始化Agent和工具]
    C --> D[设置初始状态]
    D --> E[执行工作流]
    
    %% 工作流主循环
    E --> F[思考节点Think]
    F --> G[行动节点Action]
    
    %% 决策分支
    G --> H{决策: 下一步?}
    H -->|思考| F
    H -->|工具执行| I[工具执行节点]
    H -->|结束| O[结束]
    
    %% 工具执行后的流程
    I --> J{工具执行决策}
    J -->|报告| K[报告节点]
    J -->|结束| O
    
    %% 报告后的流程
    K --> L{报告决策}
    L -->|思考| F
    L -->|结束| O
    
    %% 结束流程
    O --> M[生成最终总结]
    M --> N[返回结果]
```

## 详细工作流状态转换

```mermaid
stateDiagram-v2
    [*] --> think: 开始
    
    think --> action: 规划任务
    
    action --> think: 需要继续思考
    action --> tool_execution: 执行工具
    action --> [*]: 完成所有任务
    
    tool_execution --> report: 工具执行完成
    tool_execution --> [*]: 直接结束
    
    report --> think: 继续下一步
    report --> [*]: 工作流完成
```

## 任务执行流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant Main as main.py
    participant Agent as run_agent
    participant Think as think节点
    participant Action as action节点
    participant Tool as tool_execution节点
    participant Report as report节点
    
    User->>Main: 输入指令
    Main->>Agent: 调用run_agent()
    Agent->>Think: 初始化状态
    
    loop 任务执行循环
        Think->>Think: 分析状态和任务历史
        Think->>Think: 调用LLM生成下一步任务
        Think->>Action: 更新任务状态
        
        Action->>Action: 准备工具调用
        
        alt 有工具调用
            Action->>Tool: 执行工具
            Tool->>Tool: 调用相应工具函数
            Tool->>Report: 更新任务结果
            Report->>Think: 分析结果，继续下一步
        else 无需工具/已完成
            Action->>Think: 继续思考
        end
    end
    
    Agent->>Main: 返回最终状态
    Main->>User: 显示执行结果
```

## 状态对象结构

```mermaid
classDiagram
    class AgentState {
        +messages: List[Message]
        +task_manager: TaskManager
        +execution_status: String
        +direct_to_end: Boolean
    }
    
    class TaskManager {
        +tasks: List[TaskItem]
        +current_task_index: Integer
    }
    
    class TaskItem {
        +description: String
        +status: String
        +result: String
        +tool_name: String
        +tool_args: String
        +task_type: String
        +snapshot: String
        +extracted_data: String
        +last_execution: Boolean
        +tools_call: List[Dict]
    }
    
    AgentState *-- TaskManager
    TaskManager *-- TaskItem
```

## 使用方法

1. 确保MCP服务正在运行:`npx @playwright/mcp@latest --port 8931`
2. 运行主程序：`uv run main.py`
3. 输入指令，系统将自动规划和执行任务
4. 查看执行结果

## 示例指令
- 打开百度搜索谷歌A2A协议文章，选择最相关的2篇文章进行总结