import uvicorn
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import io
from qwen_vl_utils import process_vision_info


MODEL_PATH = "./models/Qwen3-VL-2B-Instruct" 



app = FastAPI(title="智慧农业多模态诊断平台")

# 配置 CORS (允许前端网页访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # 允许 POST, GET 等所有方法
    allow_headers=["*"],  # 允许所有请求头
)

# ================= 2. 定义专家提示词库 (论文核心创新点) =================
# 这里就是你在论文里写的“结构化专家提示词库”
PROMPT_TEMPLATES = {
    # 模式 A: 病虫害诊断专家
    "disease": (
        "你是一位经验丰富的植物病理学专家。请仔细观察这张农作物图像："
        "1. 识别图中可能存在的病虫害名称（如果不确定，请说明）；"
        "2. 分析病害的典型症状（如叶片颜色、斑点形态等）；"
        "3. 给出具体的防治建议和农药使用指导。"
        "如果图像中的作物健康，请说明生长状况良好。"
    ),
    # 模式 B: 生长监测专家
    "growth": (
        "你是一位高级农艺师。请监测图中作物的生长情况："
        "1. 判断作物当前的生长阶段（如苗期、拔节期、抽穗期、成熟期等）；"
        "2. 评估作物的长势是否健康，有无缺素症状；"
        "3. 给出当前阶段的水肥管理建议。"
    ),
    # 模式 C: 农产品分级专家
    "grading": (
        "你是一位农产品质检员。请对图中的农产品进行分级评定："
        "1. 观察果实的大小、色泽、形状和完整性；"
        "2. 根据外观特征判断其等级（特级、一级、二级或次果）；"
        "3. 指出影响等级的主要缺陷（如机械伤、虫眼、畸形等）。"
    ),
    # 模式 D: 通用助手
    "general": (
        "你是一个通用的智慧农业助手。请识别图中的内容，并回答用户的具体问题。"
    )
}

# ================= 3. 加载模型 (系统启动时运行) =================
print(f"正在加载模型权重，路径: {MODEL_PATH} ...")
try:
    # 加载模型 (自动适配显卡和精度)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(" 模型加载完毕！服务已启动，等待请求...")
except Exception as e:
    print(f" 模型加载失败，请检查路径是否正确！错误信息: {e}")
    exit(1)

# ================= 4. 核心推理接口 =================
@app.post("/analyze")
async def analyze_crop(
    file: UploadFile = File(...),   # 接收图片文件
    prompt: str = Form(""),         # 接收用户补充的问题
    task_type: str = Form("general") # 接收任务类型 (disease/growth/grading)
):
    try:
        #读取并处理图片
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))

        # B. 提示词路由逻辑 (Prompt Routing)
        # 根据 task_type 选择对应的专家模板
        system_instruction = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES["general"])
        
        # 组装最终提示词：专家指令 + 用户补充问题
        final_prompt = system_instruction
        if prompt.strip():
            final_prompt += f"\n\n用户补充提问：{prompt}"
            
        print(f"收到请求 | 任务类型: {task_type} | 最终提示词长度: {len(final_prompt)}")

        #构建模型输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": final_prompt},
                ],
            }
        ]
        
        #数据预处理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        #执行推理 (Generate)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        #解码结果
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        #返回 JSON
        return {"result": output_text, "status": "success"}

    except Exception as e:
        print(f"推理过程中出错: {str(e)}")
        return {"result": f"服务器内部错误: {str(e)}", "status": "error"}

# ================= 5. 启动入口 =================
if __name__ == "__main__":
    # 启动服务，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)