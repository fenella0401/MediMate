from flask import Flask, request, jsonify, make_response
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
from peft import PeftModel
import csv
import json
from flask_cors import CORS
from queue import Queue
import queue
import threading
import time
import random
import os
import re
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from conversations import conv_templates, get_default_conv_template, SeparatorStyle

# 加载ZhipuAI模型
from zhipuai import ZhipuAI
client = ZhipuAI(api_key='b841faf7cf70943698e2d2e14b83f971.pzAlJMN7nsxdO85g')

# model_name_or_path = '/devdata/fenella/project/llm/chatglm3-6b'
# adapter_name_or_path = '/devdata/fenella/project/llm/Firefly/output/firefly-chatglm3-6b-medical'

# # 加载base model
# model = AutoModel.from_pretrained("/devdata/fenella/project/llm/chatglm3-6b", trust_remote_code=True, device='cuda')
# # 加载adapter
# # if adapter_name_or_path is not None:
# #     model = PeftModel.from_pretrained(model, adapter_name_or_path)
# # 加载tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# model = model.eval()

# 加载pulse模型
model_name = "/devdata/fenella/project/llm/MING-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
context_len = 2048
model.config.use_cache = True
model.eval()

@torch.inference_mode()
def generate_stream(model, tokenizer, params, beam_size,
                    context_len=4096, stream_interval=2):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.2))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    input_ids = tokenizer(prompt).input_ids

    max_src_len = context_len - max_new_tokens - 8
    input_ids = torch.tensor(input_ids[-max_src_len:]).unsqueeze(0).cuda()

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=beam_size,
        temperature=temperature,
    )
    outputs = outputs[0][len(input_ids[0]):]
    output = tokenizer.decode(outputs, skip_special_tokens=True)

    return output

def pulse(history):
    '''
    history: [{'role':'user', 'content':'xxx'}, {'role':'assistant', 'content':'xxx'}]
    '''
    conv = conv_templates["bloom"].copy()
    
    for each in history:
        if each['role'] == 'user':
            conv.append_message("USER", each['content'])
        else:
            conv.append_message("ASSISTANT", each['content'])
    
    conv.append_message(conv.roles[1], None)

    generate_stream_func = generate_stream
    prompt = conv.get_prompt()

    params = {
        "prompt": prompt,
        "temperature": 1.2,
        "max_new_tokens": 512,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }

    context_len = len(prompt)  + params['max_new_tokens'] + 8
    output_stream = generate_stream_func(model, tokenizer, params, 3, context_len=context_len)
    #print(output_stream)

    return output_stream.strip()

# 开启app
app = Flask(__name__)
CORS(app)  # 允许所有跨域请求
request_queue = Queue()
res = {}

template = {
    '发热': '1.性别和年龄，2.患病时间和起病情况，3.发热的程度、规律，4.诱因（感染、炎症等），5.病情发展与演变，6.伴随症状（寒战、咽痛、关节肿痛等），7.诊疗经过、检查（血常规）、药物（抗菌类药物）、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（肺炎、结核病等）。', 
    '皮肤黏膜出血': '1.性别和年龄，2.患病时间和起病情况，3.出血的部位、颜色、量、规律、加重或缓解因素，4.诱因（外伤、虫咬、感染等），5.病情发展与演变，6.伴随症状（腹痛、鼻出血、血尿等），7.诊疗经过、检查（凝血功能、外周血检查等）、药物、疗效，8.既往史，9.月经生育史，10.家族史。', 
    '水肿': '1.性别和年龄，2.患病时间和起病情况，3.水肿的部位、程度、波动性、有无压痕、加重或缓解因素，4.诱因（饮食、药物等），5.病情发展与演变，6.伴随症状（小便疼痛、呼吸困难、心跳缓慢等），7.诊疗经过、检查（尿常规、BNP、肝肾功能、心脏超声等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（肾脏病、肝病、糖尿病等），10.个人史，11.婚姻史，12.月经生育史，13.家族史。', 
    '咳嗽与咳痰': '1.性别和年龄，2.咳嗽的性质、类型、程度、时间、加重或缓解因素，3.咳痰的性质、颜色、量、质地、时间、加重或缓解因素，4.诱因（受凉等），5.病情发展与演变，6.伴随症状（发热、流涕、呼吸困难等），7.诊疗经过、检查（血常规、胸部CT等）、药物（抗菌、止咳、祛痰药物）、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（肺炎、支气管炎、哮喘等），10.个人史（烟酒嗜好），11.家族史。', 
    '咯血': '1.性别和年龄，2.患病时间和起病情况，3.咯血的性质、颜色、规律、加重或缓解因素，4.诱因（劳累、久病等），5.病情发展与演变，6.伴随症状（发热、胸痛、咳嗽等），7.诊疗经过、检查（胸部CT、支气管镜检查等）、药物（抗生素、止血药物）、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（肺结核、支气管扩张等），10.个人史（烟酒嗜好，工作性质与环境），11.婚姻史，12.月经生育史，13.家族史。', 
    '发绀': '1.性别和年龄，2.患病时间和起病情况，3.发绀的部位、颜色，4.诱因，5.病情发展与演变，6.伴随症状（呼吸困难、意识障碍等），7.诊疗经过、检查、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（心脏病、肺部疾病等），10.个人史（烟酒嗜好、周围环境），11.月经生育史，12.家族史。',
    '呼吸困难': '1.性别和年龄，2.患病时间和起病情况，3.表现、阵发性还是持续性、加重或缓解因素，4.诱因（劳累、受凉等），5.病情发展与演变，6.伴随症状：（哮喘、发热、胸痛等），7.诊疗经过、检查（血常规、胸部CT等）、药物、疗效。8.一般情况（饮食、睡眠、体重），9.既往史（心脏病、肺部疾病等），10.个人史（烟酒嗜好，工作性质与环境），11.月经生育史，12.家族史。', 
    '胸痛': '1.性别和年龄，2.患病时间和起病情况，3.胸痛的部位、性质、程度、持续时间、有无放射性、加重或缓解因素，4.诱因（感冒、外伤、劳累等），5.病情发展与演变，6.伴随症状（咳嗽、呼吸困难咳、心悸等），7.诊疗经过、检查（胸部X线片、心电图等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（心脏病、肺部疾病、胃病等）10.个人史（烟酒嗜好，工作性质与环境），11.家族史。', 
    '心悸': '1.性别和年龄，2.患病时间和起病情况，3.心悸的表现、频率、持续时间、加重或缓解因素，4.诱因（咖啡、劳累、剧烈运动等），5.病情发展与演变，6.伴随症状（胸痛、发热、呼吸困难、消瘦等），7.诊疗经过、检查（心电图、甲状腺功能测定等）、药物（抗心律失常药物治疗）、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（心脏病、甲亢等）10.个人史（烟酒嗜好，工作性质与环境），11.家族史。', 
    '恶心与呕吐':'1.性别和年龄，2.患病时间和起病情况，3.恶心与呕吐的时间、表现、性质、呕吐物气味、性状、量、加重或缓解因素，4.诱因（饮食不当、饮酒、服用药物等），5.病情发展与演变，6.伴随症状（腹痛、发热、眩晕等），7.诊疗经过、检查（血常规、腹部B超等）、药物（止吐药、胃黏膜保护剂）、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（类似发作史、胃炎、消化性溃疡等），10.个人史（烟酒嗜好），11.月经生育史，12.家族史。', 
    '吞咽困难': '1.性别和年龄，2.患病时间和起病情况，3.吞咽困难的表现，4.诱因（饮食不当、饮酒、服用药物等），5.病情发展与演变，6.伴随症状（声嘶、呛咳、反酸等），7.诊疗经过、检查（胃镜等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（类似发作史、食管炎等），10.个人史（烟酒嗜好），11.家族史。', 
    '呕血': '1.性别和年龄，2.患病时间和起病情况，3.呕血的表现、颜色、加重或缓解因素，4.诱因（饮食不当、饮酒、服用药物、精神因素等），5.病情发展与演变，6.伴随症状（胃痛、腹痛、头晕等），7.诊疗经过、检查（血常规、胃镜、粪常规、腹部B超等）、药物（抑酸剂、止血药等）、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（类似发作史、胃病、肝病等）。10.个人史（烟酒嗜好，工作性质与环境），11.家族史。', 
    '便血': '1.性别和年龄，2.患病时间和起病情况，3.便血的颜色、量、性质、加重或缓解因素，4.诱因（饮食不当、服用药物等），5.病情发展与演变，6.伴随症状（腹痛、肛门不适等），7.诊疗经过、检查（血常规、粪常规、隐血等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（类似发作史、胃病、痔疮、肛裂等）。10.个人史（烟酒嗜好，工作性质与环境），11.家族史。', 
    '腹痛': '1.性别和年龄，2.患病时间和起病情况，3.腹痛的部位、有无转移、性质、程度、规律、有无放射性、加重或缓解因素，4.诱因（饮食不当、饮酒、外伤等），5.病情发展与演变，6.伴随症状（发热、反酸、腹泻等），7.诊疗经过、检查（血常规、胃镜、B超等）、药物（止痛药、输液等）、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（胃病、肝病、肠道疾病等），10.个人史（烟酒嗜好），11.婚姻史，12.月经生育史，13.家族史。', 
    '腹泻': '1.性别和年龄，2.患病时间和起病情况，3.每日排便次数、粪便量及性状、有无便血及脓液，4.诱因（不洁饮食、刺激性食物、劳累、服药药物等），5.病情发展与演变，6.伴随症状（发热、腹胀、腹痛等），7.诊疗经过、检查（血常规、粪常规、肠镜等）、药物（抗菌药物）、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（胃炎、肠炎等），10.个人史（烟酒嗜好、作息），11.家族史。', 
    '便秘': '1.性别和年龄，2.患病时间和起病情况，3.多久排便1次、粪便量及性状、有无费力感、肛周情况，4.诱因（进食量少、劳累、服用药物等），5.病情发展与演变，6.伴随症状（呕吐、腹胀、腹痛等），7.诊疗经过、检查（粪常规、腹部B超、肠镜等）、通便药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（糖尿病、肠易激综合征等），10.个人史（烟酒嗜好、作息），11.家族史。', 
    '黄疸': '1.性别和年龄，2.患病时间和起病情况，3.黄疸的表现、部位、程度，4.诱因（饮食不当、饮酒等），5.病情发展与演变，6.伴随症状（皮肤瘙痒、发热、腹痛等），7.诊疗经过、检查（血常规、尿常规、肝肾功能等）、药物、疗效\n2）治疗情况（是否用过糖皮质激素、保肝药物治疗，疗效，8.一般情况（饮食、睡眠、体重），9.既往史（肝炎等），10.个人史（烟酒嗜好、作息），11.家族史。', 
    '腰背痛': '1.性别和年龄，2.患病时间和起病情况，3.疼痛的具体位置、性质（酸痛、刺痛、胀痛等）、程度、持续时间、有无放射性、缓解或加剧的因素，4.诱因（外伤、久坐久站、劳累、受凉等），5.病情发展与演变，6.伴随病状（腿痛、腿麻、活动受限等），7.诊疗经过、检查（CT、腰椎X线片、核磁等）、药物（止痛药、膏药、针灸推拿等）、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（风湿病、腰肌劳损等），10.个人史（职业特点、生活习惯），11.月经生育史。', 
    '颈肩痛': '1.性别和年龄，2.患病时间和起病情况，3.疼痛具体位置、性质（酸痛、刺痛、胀痛等）、程度、持续时间、有无放射性、缓解或加剧的因素，4.诱因（不良姿势、久坐、外伤、受凉等），5.病情发展与演变，6.伴随病状（手臂麻木、活动受限等），7.诊疗经过、检查（颈椎X线片、CT、核磁等）、药物（止痛药、膏药、针灸推拿等）、疗效，8.一般情况（睡眠、运动），9.既往史（肩周炎、关节炎等），10.个人史（作息、运动），11.月经生育史。', 
    '关节痛': '1.性别和年龄，2.患病时间和起病情况，3.疼痛具体位置、性质（酸痛、刺痛、胀痛等）、程度、持续时间、缓解或加剧的因素，4.诱因（饮酒、气候、外伤等），5.病情发展与演变，6.伴随病状（红肿、压痛、晨僵等），7.诊疗经过、检查（血常规、关节X线片、核磁等）、药物（止痛药、抗菌药物），8.一般情况（睡眠、运动），9.既往史（痛风、类风湿等），10.个人史（工作性质与环境），11.月经生育史。', 
    '血尿': '1.性别和年龄，2.患病时间和起病情况，3.尿色、量、有无血凝块、是否全程血尿、呈间歇性或持续性，4.诱因（饮食等），5.病情发展与演变，6.伴随症状（尿频、尿急、尿痛、腹痛等），7.诊疗经过、检查（尿常规、肾功能等）、药物、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（尿路结石、前列腺炎等），10.个人史（烟酒嗜好），11.婚姻史，12.月经生育史，13.家族史。', 
    '尿频、尿急、尿痛': '1.性别和年龄，2.患病时间和起病情况，3.排尿频率、每次排尿量、有无尿失禁等，4.诱因（憋尿或饮水减少、精神因素等），5.病情发展与演变，6.伴随症状（尿色改变、排尿困难、腹痛等），7.诊疗经过、检查（血常规、尿常规、肾功能等）、药物、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（膀胱炎、糖尿病等），10.个人史（烟酒嗜好），11.婚姻史，12.月经生育史，13.家族史。', 
    '少尿、无尿、多尿': '1.性别和年龄，2.患病时间和起病情况，3.排尿频率、每次排尿量、有无尿失禁等，4.诱因（憋尿或饮水减少、精神因素等），5.病情发展与演变，6.伴随症状（腹痛、水肿、食欲减退等），7.诊疗经过、检查（血常规、尿常规、肾功能等）、药物、疗效，8.一般情况（饮食、睡眠、大便、体重），9.既往史（肾病、前列腺炎、糖尿病等），10.个人史（烟酒嗜好），11.婚姻史，12.月经生育史，13.家族史。', 
    '尿失禁': '1.性别和年龄，2.患病时间和起病情况，3.程度、持续性还是间歇性，4.诱因，5.病情发展与演变，6.伴随症状（多尿、消瘦、大便失禁等），7.诊疗经过、检查、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（前列腺增生、糖尿病等），10.个人史，11.婚姻史，12.月经生育史，13.家族史。', 
    '排尿困难': '1.性别和年龄，2.患病时间和起病情况，3.程度，4.诱因（精神因素等），5.病情发展与演变，6.伴随症状（尿频尿急、下腹绞痛、血尿等），7.诊疗经过、检查（B超、CT等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（前列腺增生、结石、糖尿病等），10.个人史，11.婚姻史，12.月经生育史，13.家族史。',
    '肥胖': '1.性别和年龄，2.身高、体重、腰围、持续时间，3.个人史（饮食、运动、睡眠等），4.伴随症状（糖尿病、月经不调等），5.诊疗经过、药物、疗效，6.既往史（糖尿病、多囊等），7.婚姻史，8.月经生育史，9.家族史。',
    '消瘦': '1.性别和年龄，2.身高、体重、体重减轻的程度和速度，3.诱因（精神因素、劳累、服用药物），4.伴随症状（腹痛、腹泻、多尿等），5.诊疗经过、检查（血常规、尿常规、尿糖、血糖、糖耐量试验、甲状腺功能）、药物、疗效，6.一般情况（饮食、睡眠），7.既往史（胃病、糖尿病等），8.个人史（烟酒嗜好、工作强度），9.婚姻史，10.月经生育史，11.家族史（糖尿病）', 
    '头痛': '1.性别和年龄，2.患病时间和起病情况，3.头痛的部位、性质、程度、规律、加重或缓解因素，4.诱因（受凉、外伤、精神因素等），5.病情发展与演变，6.伴随症状（呕吐、视力障碍等），7.诊疗经过、检查（头颅CT、核磁共振等）、药物（止痛药）、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（高血压、血管堵塞等），10.个人史（烟酒嗜好，工作性质与环境），11.月经生育史，12.家族史。', 
    '眩晕': '1.性别和年龄，2.患病时间和起病情况，3.眩晕的表现、程度、规律、加重或缓解因素，4.诱因（劳累、服用药物、精神因素等），5.病情发展与演变，6.伴随症状（耳鸣、恶心、听力下降、站立或行走不稳等），7.诊疗经过、检查（血常规、头颅CT等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（晕动病、贫血、高血压等），10.个人史（烟酒嗜好，工作性质与环境），11.月经生育史，12.家族史。', 
    '晕厥': '1.性别和年龄，2.晕厥的次数、时间、持续时间，3.诱因（情绪激动、低血糖、劳累等），4.伴随症状（头痛、出冷汗等），5.诊疗经过、检查（血常规、头颅CT等）、药物、疗效，6.一般情况（饮食、睡眠、体重），7.既往史（晕动病、贫血、高血压等），8.个人史（烟酒嗜好，工作性质与环境），9.月经生育史，10.家族史。',
    '抽搐与惊厥': '1.性别和年龄，2.抽搐的表现、部位、时间、持续时间、前驱症状，3.诱因（情绪激动、特殊环境等），4.伴随症状（发热、剧烈头痛等），5.诊疗经过、检查（脑CT、心电图等）、药物（抗癫痫药物、止惊药物）、疗效，6.一般情况（饮食、睡眠、体重），7.既往史（高血压、肾炎、癫痫等），8.个人史（工作性质与环境、饮食习惯），9.月经生育史，10.家族史。', 
    '意识障碍': '1.性别和年龄，2.患病时间和起病情况，3.具体表现、时间、持续时间，4.诱因（中毒、中暑、剧烈运动等），5.病情发展与演变，6.伴随症状（发热、呼吸缓慢、瞳孔散大等），7.诊疗经过、检查（脑CT、心电图等）、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史（高血压、肾炎等），10.个人史（烟酒嗜好、工作性质与环境），11.家族史。', 
    '情感症状': '1.性别和年龄，2.患病时间和起病情况，3.具体表现、周期性、季节性，4.诱因（特殊事件、精神因素等），5.病情发展与演变，6.伴随症状（反应速度、注意力降低、乏力等），7.诊疗经过、检查、药物、疗效，8.一般情况（饮食、睡眠、体重），9.既往史，10.个人史（工作性质与环境、生活压力、病前性格），11.婚姻史，12.月经生育史，13.家族史（精神障碍、焦虑症）。',
}

template_all = ['发热','皮肤黏膜出血','水肿','咳嗽与咳痰','咯血','发绀','呼吸困难','胸痛','心悸','恶心与呕吐','吞咽困难','呕血','便血','腹痛','腹泻','便秘','黄疸','腰背痛','颈肩痛','关节痛','血尿，尿频、尿急、尿痛','少尿、无尿、多尿','尿失禁','排尿困难','肥胖','消瘦','头痛','眩晕','晕厥','抽搐与惊厥','意识障碍','情感症状']

example = "\n\n以下是一个优秀问诊案例：\n患者：我上腹疼，想看看究竟是什么病，是不是得癌症了？\n医生：您好，我是内科门诊医生。您不用着急，我先好好了解一下您的病情，希望您能合作。方便告诉我您的性别和年龄？\n患者：男，33岁。\n医生：您上腹疼多长时间了？\n患者：半个月前一直疼到现在还在疼。\n医生：您能讲一讲都怎么疼法的么？比如是一阵阵的疼还是一个劲的疼？是针扎样疼还是刀割样疼，拧劲疼 顶着顶着疼还是丝丝拉拉的疼 能忍受不？\n患者：是一个劲的疼，能忍受。\n医生：疼时往哪个地方窜不？\n患者：往右后背窜疼。\n医生：您能讲一讲哪些因素使您上腹疼痛加剧么？\n患者：快要饿的时候疼的重，夜间有时疼醒，再有吃油茶等油性的的东西的时候疼的重。\n医生：哪些因素有能使您的上腹疼得轻点呢？\n患者：吃点饭或吃点饼干，我兜里总装些饼干，疼就吃点，吃点小苏打饼干疼就轻多了，但还一直疼。\n医生：您再讲一下上腹疼与吃饭有没有关系？是吃饱了疼还是饿了疼？\n患者：快饿的时候疼，吃点东西就减轻了，但吃油大的食物还是不行。\n医生：您能想起来引起上腹疼的原因么？\n患者：这次疼是因为我连续两天一宿没休息开车累着了，就丝丝拉拉的疼了。\n医生：这半个月疼痛感有什么变化吗？\n患者：感觉有点加重。\n医生：除了上腹疼还有哪个地方不舒服么？\n患者：有！除了上腹疼以外，还经常烧心、打嗝、恶心、吐过一次酸水，还带点饭。\n医生：吐的是当顿的饭么？有血没有？\n患者：是当天吃的饭，就吐了两口，没有血。\n医生：您大便好不好？\n患者：大便干，色黑，每天一次，量不多，昨天排一次像柏油马路那样的黑色便，只有两盅，今天没便。\n医生：小便正常不？尿黄不？有人发现您眼睛黄不？\n患者：尿正常。没人说我眼睛黄。\n医生：您觉得发烧不？\n患者：不觉得热。\n医生：排黑便后您觉得有没有心跳、头迷，眼花、出汗？\n患者：没有。心不跳，眼不花。没出汗。\n医生：您有去医院做过什么检查吗？有没有服用什么药物？\n患者：没去医院，也没吃药。\n医生：您病后食欲怎么样？瘦没？睡眠好么？\n患者：我病后爱吃饭，没见瘦。以前睡觉好，这半个月经常因为心口疼醒，吃点饼干照样睡。\n医生：得病后影响工作没？\n患者：没影响工作，照样开车，没办法我就用小苏打顶着。\n医生：您讲的很全面，以上我了解了您这次得病的情况，为了诊断，我想了解一下您过去的身体情况，总的来说怎么样？得过什么大病没有？\n患者：三年曾有一段时间胃疼，也是这个位置。有一个多星期，吃胃药才好。没到医院检查。1年前体检B超发现胆石症，未经特殊治疗。\n医生：您抽烟么？\n患者：抽。一天十多支。有四年多了。\n医生：喝酒不？\n患者：有时喝点啤酒，没喝过白酒。\n医生：开汽车多少年了？生活规律不？\n患者：开汽车已经10年了。早出晚归的，吃饭也不准时，生活不规律。\n医生：您结婚没有？\n患者：还没结婚呢。\n医生：您家里人得过什么病没？有没有得这样病的？\n患者：父亲得过高血压病。\n医生：好的，您的情况我了解了，初步诊断您可能是慢性胆囊炎、上消化道出血或胆石症，这些疾病都会引起上腹疼、恶心、呕吐等症状。建议您还是要先尽快到医院消化内科做个胃镜检查，明确诊断，才能针对性治疗。确定诊断后您可以按照医生的治疗建议进行治疗，也可以再来找我咨询，将检查结果告知我，我结合您的情况再为您提供治疗建议。您不用着急，胃癌的可能性应该不大，您尽快检查即可。"

prompt_zdx1 = '问诊病史采集信息内容如下：\n"""1.性别和年龄，2.患病时间和起病情况，3.主要症状的部位、性质、持续时间、程度、缓解或加剧的因素，4.病因和诱因，5.病情发展与演变，6.伴随病状，7.诊疗经过、检查、药物、疗效，8.病程中的一般情况，9.既往史，10.个人史，11.婚姻史，12.月经生育史，13.家族史。"""\n\n'
prompt_zdx2 = '”症状病史采集问诊内容如下：\n"""'
prompt_zdx3 = '"""\n\n'
prompt_role1 = '假设你是一个经验丰富、非常谨慎、亲切、有礼貌、有耐心的'
prompt_role2 = '临床医生。'
prompt_probe = '你要根据问诊学思路，分析回答我的健康问题需要了解哪些病史信息，并有逻辑地逐个询问我必要的病史信息，简述询问理由。切记每次仅询问1-3个问题，直至询问完成，内容尽量精炼简短，询问过程中不用说明诊断！采集完必要的信息后给出诊断结果与建议，格式为“诊断：xx\n建议：xx”。'
prompt_anaylsis1 = '你需要根据以下对话：\n"""'
prompt_anaylsis2 = '""" \n首先梳理我的病史信息，然后综合分析我的病史，回答我的提问："""'
prompt_anaylsis3 = '"""。回答内容包括：1.病史梳理，2.病情诊断（综合病史给出最可能的诊断，说明诊断原因；同时简述其他可能的情况），3.诊疗建议（推荐就医科室、就医紧急程度、推荐检查、初步治疗方法），4.生活习惯建议。'

def llm(data):
    print('data:', data)
    try_num = 0

    while True:
        if try_num > 2:
            result = {'user_id': data['user_id'], 'response': '很抱歉，系统出现问题，给您带来不愉快的体验，请您重新输入', 'history': data['history'], 'status': data['status'], 'category': data['category']}
            return result
        else:
            try_num += 1
            try:
                index = data['user_id']
                if index == 0:
                    file_num = len(os.listdir('./user_records_1/'))
                    index = file_num + 1
                    
                # 体验第一个系统
                if data['status'] != '4':
                    model_name = 'Our-GLM4'
                    # 首次请求
                    if data['history'] == []:
                        system_response = client.chat.completions.create(model="glm-4",messages=[{"role": "user", "content": '判断“'+data['user']+'”是否为健康相关问题。直接告诉我“是”或“否”，不要生成其他的内容。'}])
                        print('是相关问题吗', system_response.choices[0].message.content)
                        if '是' in system_response.choices[0].message.content:       
                            system_response = client.chat.completions.create(model="glm-4",messages=[{"role": "user", "content": '医院科室分类：\n"""1.预防保健科，2.全科，3.呼吸内科，4.消化内科，5.神经内科，6.心血管内科，7.血液内科，8.肾病学，9.内分泌，10.免疫学，11.变态反应，12.老年病，13.普通内科，14.普通外科，15.肝脏移植，16.胰腺移植，17.小肠移植，18.神经外科，19.骨科，20.泌尿外科，21.肾病移植，22.胸外科，23.肺脏移植，24.心脏大血管外科，25.心脏移植，26.烧伤外科，27.整形外科，28.介入科，29.妇科，30.产科，31.计划生育，32.优生学，33.生殖健康与不孕症，34.妇产科普通，35.妇女保健科，36.儿科，37.小儿外科，38.儿童保健科，39.眼科，40.耳鼻咽喉科，41.口腔科，42.皮肤科，43.医疗美容科，44.精神科，45.传染科，46.中医科"""\n患者提问：\n"""'+data['user']+'"""\n假设你是一个专业的护士，根据患者提问，建议患者最应该就诊的一个科室，提取科室序号，以JSON格式输出{"department": xx}。'}])
                            print('科室：', system_response.choices[0].message.content)
                            department_all = ['预防保健科', '全科', '呼吸内科', '消化内科', '神经内科', '心血管内科', '血液内科', '肾病学', '内分泌', '免疫学', '变态反应', '老年病', '普通内科', '普通外科', '肝脏移植', '胰腺移植', '小肠移植', '神经外科', '骨科', '泌尿外科', '肾病移植', '胸外科', '肺脏移植', '心脏大血管外科', '心脏移植', '烧伤外科', '整形外科', '介入科', '妇科', '产科', '计划生育', '优生学', '生殖健康与不孕症', '妇产科普通', '妇女保健科', '儿科', '小儿外科', '儿童保健科', '眼科', '耳鼻咽喉科', '口腔科', '皮肤科', '医疗美容科', '精神科', '传染科', '中医科']
                            department = re.findall(r'{.*?}', system_response.choices[0].message.content.replace('\n', ''))
                            department = json.loads(department[0])
                            department = department['department']
                            category = department_all[department-1]
                            print(category)

                            system_response = client.chat.completions.create(model="glm-4",messages=[{"role": "user", "content": '临床医学中32种常见症状如下：\n"""1.发热，2.皮肤黏膜出血，3.水肿，4.咳嗽与咳痰，5.咯血，6.发绀，7.呼吸困难，8.胸痛，9.心悸，10.恶心与呕吐，11.吞咽困难，12.呕血，13.便血，14.腹痛，15.腹泻，16.便秘，17.黄疸，18.腰背痛，19.颈肩痛，20.关节痛，21.血尿，尿频、尿急、尿痛，22.少尿、无尿、多尿，23.尿失禁，24.排尿困难，25.肥胖，26.消瘦，27.头痛，28.眩晕，29.晕厥，30.抽搐与惊厥，31.意识障碍，32.情感症状"""\n患者提问：\n"""'+data['user']+'"""\n请分析患者主要患有上述哪一症状，提取症状序号，以JSON格式输出{"symptom_numbers": [xx]}。如果没有对应症状，请输出{"symptom_numbers": []}。'}])
                            print('症状：', system_response.choices[0].message.content)
                            symptom = re.findall(r'{.*?}', system_response.choices[0].message.content.replace('\n', ''))
                            symptom = json.loads(symptom[0])
                            symptom_list = symptom['symptom_numbers']
                            
                            if symptom_list == []:
                                symptom = ''
                                system_prompt = prompt_zdx1 + prompt_role1 + category + prompt_role2 + prompt_probe
                            else:
                                system_prompt = ''
                                symptom = ''
                                each = symptom_list[0]
                                symptom += template_all[each-1] + ' '
                                system_prompt = system_prompt + '“' + template_all[each-1] + prompt_zdx2 + template[template_all[each-1]] + prompt_zdx3
                                system_prompt += prompt_role1 + category + prompt_role2 + prompt_probe
                            
                            print(symptom)

                            #system_prompt = prompt_zdx1 + prompt_role1 + category + prompt_role2 + prompt_probe
                            #system_prompt = prompt_role1 + category + prompt_role2 + prompt_probe
                            response = '4 四'
                            while '4' in response or '四' in response or ('诊断：' in response and '建议：' in response):
                                print('error:', response)
                                tempreture = round(random.uniform(0.4, 0.8), 1)
                                print(tempreture)
                                response = client.chat.completions.create(model="glm-4",messages=[{'content': system_prompt, 'role': 'system'}, {'content': data['user'], 'role': 'user'}],temperature=tempreture)
                                response = response.choices[0].message.content

                            response = '您好，感谢您的耐心等候，我是智能健康咨询机器人医生。' + response
                            history = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': data['user']}, {'role': 'assistant', 'content': response}]
                            
                            result = {'user_id': index, 'response': response, 'history': history, 'status': '2', 'category': category}
                            record_data = {'user_id': index, 'user_question': data['user'], model_name: {'history': history, 'status': '2', 'category': category, 'symptom': symptom, 'starttime': data['start_time']}}
                            with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                json.dump(record_data, f, indent=4, ensure_ascii=False)
                            return result
                        
                        else:
                            # 非健康相关问题
                            result = {'response': '很抱歉，您问的不是健康问题，请您直接说出您想咨询的健康问题，例如“我左腰后侧酸痛，怎么回事怎么办？”', 'user_id': 0, 'history': [], 'status': '2', 'category': ''}
                            return result
                    
                    # 非第一次请求
                    else:
                        history = data['history']
                        history.append({'role': 'user', 'content': data['user']})
                        response = client.chat.completions.create(model="glm-4",messages=history)
                        response = response.choices[0].message.content
                        
                        while ('?' in response or '？' in response) and ('诊断：' in response or '建议：' in response):
                            temperature = round(random.uniform(0.4, 0.8), 1)
                            response = client.chat.completions.create(model="glm-4",messages=history,temperature=temperature)
                            response = response.choices[0].message.content
                        history.append({'role': 'assistant', 'content': response})
                        
                        if ('诊断：' in response or '建议：' in response) and ('?' not in response and '？' not in response):
                            if_ask_prompt = [
                                    {
                                        "role": "user",
                                        "content": "以下是一段医生给患者的回复：\n\"\"\"\n" + response + "\n\"\"\"\n\n请问医生是否向患者询问了更多的信息，仅回答“是”或“否”，不要生成其他内容。"
                                    },
                                ]
                            if_ask_response = client.chat.completions.create(model="glm-4",messages=if_ask_prompt)
                            if_ask_response = if_ask_response.choices[0].message.content
                            print('OUR:', if_ask_response)

                            if '否' in if_ask_response:
                                prompt_history = ''
                                for i, each in enumerate(history[:-1]):
                                    if each['role'] == 'user':
                                        prompt_history += '我：' + each['content'] + '\n'
                                    if each['role'] == 'assistant':
                                        prompt_history += '医生：' + each['content'] + '\n'
                                
                                messages = [{'role':'user', 'content': prompt_role1 + data['category'] + prompt_role2 + prompt_anaylsis1+prompt_history+prompt_anaylsis2+history[1]['content']+prompt_anaylsis3}]
                                response_init = client.chat.completions.create(model="glm-4",messages=messages)
                                response_init = response_init.choices[0].message.content
                                #print('初步诊断:', response_init)

                                prompt_history2 = history[1:-1]
                                prompt_history2.append({'role': 'assistant', 'content': response_init})
                                #prompt_history2.append({'role': 'user', 'content': '基于当前您病情诊断的几种诊断，还能否进一步询问我2-3个具体问题快速排除概率较小的情况？如果可以请直接询问我，以“感谢您的描述。为了进一步明确诊断，请问”类似句子为开头，注意不要询问之前询问过的问题；否则请直接回复“初步诊断已明确，请线下就医”，不要生成其他内容。'})
                                prompt_history2.append({'role': 'user', 'content': '对于您当前的病情诊断，是否有不把握的地方需要进一步询问我病史信息的？若是请直接询问我，以“感谢您的描述。为了进一步明确诊断，请问”类似句子为开头，注意不要询问之前询问过的问题；否则请直接回复“初步诊断已明确，请线下就医”，不要生成其他内容。'})
                                if_ask_more = client.chat.completions.create(model="glm-4",messages=prompt_history2)
                                if_ask_more = if_ask_more.choices[0].message.content
                                print('是否进一步询问:', if_ask_more)
                                
                                if '初步诊断已明确' in if_ask_more:
                                    # 专家组讨论
                                    # 你是一个经验丰富的临床医生。以下是一份诊断记录，你需要找出其中病情诊断、诊疗建议或生活习惯建议中的错误，并说明。
                                    # messages = [
                                    #         {
                                    #             "role": "user",
                                    #             "content": "以下是一份诊断记录：\n\"\"\"\n" + response_init + "\n\"\"\"\n\n你是一位经验丰富的临床医生。请你找出该诊断记录中病情诊断、诊疗建议或生活习惯建议中的错误，并说明。"
                                    #         },
                                    #     ]
                                    messages.append({"role": "assistant", "content": response_init})
                                    messages.append({"role": "user", "content": "你是团队中另一位经验丰富的临床医生。请你找出前面医生给出的病情诊断、诊疗建议或生活习惯建议中的错误，并说明。"})
                                    response = client.chat.completions.create(model="glm-4",messages=messages)

                                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                                    messages.append({"role": "user", "content": "你是另一位经验丰富的临床医生。基于前面两位医生的想法，请问你还有什么不同的或补充的意见吗？"})
                                    response = client.chat.completions.create(model="glm-4",messages=messages)

                                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                                    messages.append({"role": "user", "content": "您是一位专家大夫。基于诊断记录和前面三位医生的讨论，请您做出专业的判断或补充，并为患者重新生成问诊回复，内容包括：1.病史梳理，2.病情诊断（综合病史给出最可能的诊断，说明诊断原因；同时简述其他可能的情况），3.诊疗建议（推荐就医科室、就医紧急程度、推荐检查、初步治疗方法），4.生活习惯建议。并以以下内容为开头：“您好，感谢您的耐心等候。根据您的情况，智能健康咨询机器人医生团队进行了充分讨论，最终诊断与建议如下：\n”"})
                                    response = client.chat.completions.create(model="glm-4",messages=messages)

                                    messages.append({"role": "assistant", "content": response.choices[0].message.content})
                                    history_result = messages
                                    
                                    result = {'user_id': index, 'response': response.choices[0].message.content, 'history': [], 'status': '3', 'category': data['category']}
                                    with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                        record_data = json.load(f)
                                    record_data[model_name]['history'] = history
                                    record_data[model_name]['history_result'] = history_result
                                    record_data[model_name]['endtime'] = time.time()
                                    record_data[model_name]['status'] = '3'
                                    with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                        json.dump(record_data, f, indent=4, ensure_ascii=False)
                                    return result
                                else:
                                    # 进一步询问
                                    history_new = history[:-1]
                                    history_new.append({'role': 'assistant', 'content': if_ask_more})
                                    result = {'user_id': index, 'response': if_ask_more, 'history': history_new, 'status': '2', 'category': data['category']}
                                    with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                        record_data = json.load(f)
                                    record_data[model_name]['history'] = history_new
                                    with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                        json.dump(record_data, f, indent=4, ensure_ascii=False)
                                    return result
                            else:
                                # 未生成诊断
                                result = {'user_id': index, 'response': response, 'history': history, 'status': '2', 'category': data['category']}
                                with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                    record_data = json.load(f)
                                record_data[model_name]['history'] = history
                                with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                    json.dump(record_data, f, indent=4, ensure_ascii=False)
                                return result
                        else:
                            # 未生成诊断
                            result = {'user_id': index, 'response': response, 'history': history, 'status': '2', 'category': data['category']}
                            with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                record_data = json.load(f)
                            record_data[model_name]['history'] = history
                            with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                json.dump(record_data, f, indent=4, ensure_ascii=False)
                            return result
                
                # 体验第二个系统 status=4 前一系统已体验结束，这里结束可以标识为status=5
                else:
                    model_name = 'XR1'
                    # 首次请求
                    if data['history'] == []:
                        # 判断是否和上一问题一样
                        with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                            record_data = json.load(f)
                        
                        # 结束模型A对话
                        if data['user'] == '确定':
                            result = {'response': '### 感谢您耐心体验人工智能模型A提供的健康咨询服务。下面由人工智能模型B与您进行对话，请您再次在对话框输入您刚刚询问模型A的健康问题。', 'user_id': data['user_id'], 'history': [], 'status': data['status'], 'category': data['category']}
                            return result
                        
                        # 进行模型B对话
                        if data['user'] == record_data['user_question']:
                            system_response = client.chat.completions.create(model="glm-4",messages=[{"role": "user", "content": '医院科室分类：\n"""1.预防保健科，2.全科，3.呼吸内科，4.消化内科，5.神经内科，6.心血管内科，7.血液内科，8.肾病学，9.内分泌，10.免疫学，11.变态反应，12.老年病，13.普通内科，14.普通外科，15.肝脏移植，16.胰腺移植，17.小肠移植，18.神经外科，19.骨科，20.泌尿外科，21.肾病移植，22.胸外科，23.肺脏移植，24.心脏大血管外科，25.心脏移植，26.烧伤外科，27.整形外科，28.介入科，29.妇科，30.产科，31.计划生育，32.优生学，33.生殖健康与不孕症，34.妇产科普通，35.妇女保健科，36.儿科，37.小儿外科，38.儿童保健科，39.眼科，40.耳鼻咽喉科，41.口腔科，42.皮肤科，43.医疗美容科，44.精神科，45.传染科，46.中医科"""\n患者提问：\n"""'+data['user']+'"""\n假设你是一个专业的护士，根据患者提问，建议患者最应该就诊的一个科室，提取科室序号，以JSON格式输出{"department": xx}。'}])
                            print('科室：', system_response.choices[0].message.content)
                            department_all = ['预防保健科', '全科', '呼吸内科', '消化内科', '神经内科', '心血管内科', '血液内科', '肾病学', '内分泌', '免疫学', '变态反应', '老年病', '普通内科', '普通外科', '肝脏移植', '胰腺移植', '小肠移植', '神经外科', '骨科', '泌尿外科', '肾病移植', '胸外科', '肺脏移植', '心脏大血管外科', '心脏移植', '烧伤外科', '整形外科', '介入科', '妇科', '产科', '计划生育', '优生学', '生殖健康与不孕症', '妇产科普通', '妇女保健科', '儿科', '小儿外科', '儿童保健科', '眼科', '耳鼻咽喉科', '口腔科', '皮肤科', '医疗美容科', '精神科', '传染科', '中医科']
                            department = re.findall(r'{.*?}', system_response.choices[0].message.content.replace('\n', ''))
                            department = json.loads(department[0])
                            department = department['department']
                            category = department_all[department-1]
                            print(category)

                            system_response = client.chat.completions.create(model="glm-4",messages=[{"role": "user", "content": '临床医学中32种常见症状如下：\n"""1.发热，2.皮肤黏膜出血，3.水肿，4.咳嗽与咳痰，5.咯血，6.发绀，7.呼吸困难，8.胸痛，9.心悸，10.恶心与呕吐，11.吞咽困难，12.呕血，13.便血，14.腹痛，15.腹泻，16.便秘，17.黄疸，18.腰背痛，19.颈肩痛，20.关节痛，21.血尿，尿频、尿急、尿痛，22.少尿、无尿、多尿，23.尿失禁，24.排尿困难，25.肥胖，26.消瘦，27.头痛，28.眩晕，29.晕厥，30.抽搐与惊厥，31.意识障碍，32.情感症状"""\n患者提问：\n"""'+data['user']+'"""\n请分析患者主要患有上述哪一症状，提取症状序号，以JSON格式输出{"symptom_numbers": [xx]}。如果没有对应症状，请输出{"symptom_numbers": []}。'}])
                            print('症状：', system_response.choices[0].message.content)
                            symptom = re.findall(r'{.*?}', system_response.choices[0].message.content.replace('\n', ''))
                            symptom = json.loads(symptom[0])
                            symptom_list = symptom['symptom_numbers']
                            
                            if symptom_list == []:
                                symptom = ''
                                system_prompt = prompt_zdx1 + prompt_role1 + category + prompt_role2 + prompt_probe
                            else:
                                system_prompt = ''
                                symptom = ''
                                each = symptom_list[0]
                                symptom += template_all[each-1] + ' '
                                system_prompt = system_prompt + '“' + template_all[each-1] + prompt_zdx2 + template[template_all[each-1]] + prompt_zdx3
                                system_prompt += prompt_role1 + category + prompt_role2 + prompt_probe
                            
                            print(symptom)

                            system_prompt = '假设你是一位经验丰富并且非常谨慎的的医生，会通过和患者进行多次的问答来明确自己的猜测，并且每次只能提一个问题，最终给出诊断结果与建议。'

                            tempreture = round(random.uniform(0.4, 0.8), 1)
                            print(tempreture)
                            response = client.chat.completions.create(model="glm-4",messages=[{'content': system_prompt, 'role': 'system'}, {'content': data['user'], 'role': 'user'}],temperature=tempreture)
                            response = response.choices[0].message.content

                            response = '您好，感谢您的耐心等候，我是智能健康咨询机器人医生。' + response
                            history = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': data['user']}, {'role': 'assistant', 'content': response}]
                            
                            result = {'user_id': index, 'response': response, 'history': history, 'status': '4', 'category': data['category']}
                            with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                record_data = json.load(f)
                            record_data[model_name] = {'history': history, 'status': '4', 'category': data['category'], 'starttime': data['start_time'], 'symptom': ''}
                            with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                json.dump(record_data, f, indent=4, ensure_ascii=False)
                            return result
                        else:
                            # 非一致问题
                            result = {'response': '很抱歉，您询问的健康问题与您询问上一模型的内容不同，请您重新输入或直接复制粘贴。', 'user_id': data['user_id'], 'history': [], 'status': data['status'], 'category': data['category']}
                            return result
                    
                    # 非第一次请求
                    else:
                        if data['user'] == '确定':
                            result = {'response': '### 感谢您耐心体验人工智能模型B提供的健康咨询服务。\n### 您的用户ID是'+str(data['user_id'])+'，请您记住该ID，返回“见数”平台填写问卷。注意请勿关闭本界面，填写问卷时需要对照两个模型与您的对话细节对两个模型提供的智能健康咨询服务进行评分。\n注：实际您可以在系统生成健康诊断与建议后继续与系统进行对话，为节省您的时间，因此仅请您体验一轮健康问答。系统生成的健康建议请咨询医生谨慎评估后再采纳，再次感谢您认真体验系统以完成本实验！', 'user_id': data['user_id'], 'history': data['history'], 'status': data['status'], 'category': data['category']}
                            return result
                        
                        history = data['history']
                        history.append({'role': 'user', 'content': data['user']})
                        response = client.chat.completions.create(model="glm-4",messages=history)
                        response = response.choices[0].message.content
                        
                        # while ('?' in response or '？' in response) and ('诊断：' in response or '建议：' in response):
                        #     temperature = round(random.uniform(0.4, 0.8), 1)
                        #     response = client.chat.completions.create(model="glm-4",messages=history,temperature=temperature)
                        #     response = response.choices[0].message.content
                        history.append({'role': 'assistant', 'content': response})
                        
                        if '?' not in response and '？' not in response:
                            if_ask_prompt = [
                                    {
                                        "role": "user",
                                        "content": "以下是一段医生给患者的回复：\n\"\"\"\n" + response + "\n\"\"\"\n\n请问医生是否向患者询问了更多的信息，仅回答“是”或“否”，不要生成其他内容。"
                                    },
                                ]
                            if_ask_response = client.chat.completions.create(model="glm-4",messages=if_ask_prompt)
                            if_ask_response = if_ask_response.choices[0].message.content
                            print('XR:', if_ask_response)

                            if '否' in if_ask_response:
                                prompt_history = ''
                                for i, each in enumerate(history[:-1]):
                                    if each['role'] == 'user':
                                        prompt_history += '我：' + each['content'] + '\n'
                                    if each['role'] == 'assistant':
                                        prompt_history += '医生：' + each['content'] + '\n'
                                
                                messages = [{'role':'user', 'content': prompt_role1 + data['category'] + prompt_role2 + prompt_anaylsis1+prompt_history+prompt_anaylsis2+history[1]['content']+prompt_anaylsis3}]
                                response_init = client.chat.completions.create(model="glm-4",messages=messages)
                                response_init = response_init.choices[0].message.content
                                #print('初步诊断:', response_init)

                                # 专家组讨论
                                # 你是一个经验丰富的临床医生。以下是一份诊断记录，你需要找出其中病情诊断、诊疗建议或生活习惯建议中的错误，并说明。
                                # messages = [
                                #         {
                                #             "role": "user",
                                #             "content": "以下是一份诊断记录：\n\"\"\"\n" + response_init + "\n\"\"\"\n\n你是一位经验丰富的临床医生。请你找出该诊断记录中病情诊断、诊疗建议或生活习惯建议中的错误，并说明。"
                                #         },
                                #     ]
                                messages.append({"role": "assistant", "content": response_init})
                                messages.append({"role": "user", "content": "你是团队中另一位经验丰富的临床医生。请你找出前面医生给出的病情诊断、诊疗建议或生活习惯建议中的错误，并说明。"})
                                response = client.chat.completions.create(model="glm-4",messages=messages)

                                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                                messages.append({"role": "user", "content": "你是另一位经验丰富的临床医生。基于前面两位医生的想法，请问你还有什么不同的或补充的意见吗？"})
                                response = client.chat.completions.create(model="glm-4",messages=messages)

                                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                                messages.append({"role": "user", "content": "您是一位专家大夫。基于诊断记录和前面三位医生的讨论，请您做出专业的判断或补充，并为患者重新生成问诊回复，内容包括：1.病史梳理，2.病情诊断（综合病史给出最可能的诊断，说明诊断原因；同时简述其他可能的情况），3.诊疗建议（推荐就医科室、就医紧急程度、推荐检查、初步治疗方法），4.生活习惯建议。并以以下内容为开头：“您好，感谢您的耐心等候。根据您的情况，智能健康咨询机器人医生团队进行了充分讨论，最终诊断与建议如下：\n”"})
                                response = client.chat.completions.create(model="glm-4",messages=messages)

                                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                                history_result = messages
                                
                                result = {'user_id': index, 'response': response.choices[0].message.content, 'history': history, 'status': '3', 'category': data['category']}
                                with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                    record_data = json.load(f)
                                record_data[model_name]['history'] = history
                                record_data[model_name]['history_result'] = history_result
                                record_data[model_name]['endtime'] = time.time()
                                record_data[model_name]['status'] = '3'
                                with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                    json.dump(record_data, f, indent=4, ensure_ascii=False)
                                return result
                            else:
                                # 未生成诊断
                                result = {'user_id': index, 'response': response, 'history': history, 'status': '4', 'category': data['category']}
                                with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                    record_data = json.load(f)
                                record_data[model_name]['history'] = history
                                with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                    json.dump(record_data, f, indent=4, ensure_ascii=False)
                                return result
                        else:
                            # 未生成诊断
                            result = {'user_id': index, 'response': response, 'history': history, 'status': '4', 'category': data['category']}
                            with open('./user_records_1/'+str(index)+'.json', 'r', encoding='utf-8') as f:
                                record_data = json.load(f)
                            record_data[model_name]['history'] = history
                            with open('./user_records_1/'+str(index)+'.json', 'w', encoding='utf-8') as f:
                                json.dump(record_data, f, indent=4, ensure_ascii=False)
                            return result
            except Exception as e:
                print(e)

# 工作线程函数，它从队列中取出请求并处理它们
def worker():
    while True:
        try:
            # 从队列中获取请求
            data = request_queue.get()
            
            # 处理请求
            result = llm(data)
            res[data['request_id']] = result
            
            # 标记任务完成
            request_queue.task_done()
        
        except queue.Empty:
            # 如果队列为空，那么线程将等待新的请求
            break

@app.route('/chat', methods=['POST'])
def api_chat():
    # 从请求中获取参数
    data = request.json

    # 生成一个唯一的请求标识符
    request_id = str(time.time())

    # 将请求数据和请求标识符放入队列
    data['request_id'] = request_id
    request_queue.put(data)

    return jsonify({'request_id': request_id})

@app.route('/get_tasks', methods=['POST'])
def api_gettask():
    # 从请求中获取参数
    data = request.json

    if data['request_id'] in res.keys():
        result = res[data['request_id']]
        del res[data['request_id']]
        return jsonify(result)
    else:
        result = {'status': '1'}
        return jsonify(result)
    
@app.route('/submit', methods=['POST'])
def api_submit():
    # 从请求中获取参数
    data = request.json

    with open('./user_records_1/'+str(data['user_id'])+'.json', 'r', encoding='utf-8') as f:
        record_data = json.load(f)
    record_data['submit'] = data['submit']
    with open('./user_records_1/'+str(data['user_id'])+'.json', 'w', encoding='utf-8') as f:
        json.dump(record_data, f, indent=4, ensure_ascii=False)

    # with open('/devdata/fenella/project/llm/api_sy_4_8.tsv', 'a', encoding='utf-8') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerow([data['start_time'], data['user_id'], data['submit']])

    return jsonify({'status': '4'})

if __name__ == '__main__':
    # 启动工作线程
    worker_thread = threading.Thread(target=worker)
    worker_thread.daemon = True
    worker_thread.start()

    app.run(host='0.0.0.0', port=5000)