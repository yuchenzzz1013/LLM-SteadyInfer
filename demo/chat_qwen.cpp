#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "model/qwen2.h"

// 聊天消息结构体
struct ChatMessage {
  std::string role;     // 角色: system, user, assistant
  std::string content;  // 内容
};

// 生成配置结构体
struct GenerationConfig {
  int max_length = 1024;           // 最大生成长度
  int max_context_length = 20480;  // 最大上下文长度
};

// 聊天助手类
class ChatAssistant {
 public:
  ChatAssistant(const std::string& model_path, const std::string& tokenizer_path)
      : model_path_(model_path), tokenizer_path_(tokenizer_path) {}

  // 初始化模型
  bool init() {
    try {
      model_ = std::make_unique<model::Qwen2Model>(base::TokenizerType::kEncodeBpe, tokenizer_path_,
                                                   model_path_, false);

      auto init_status = model_->init(base::DeviceType::kDeviceCUDA);
      if (!init_status) {
        LOG(ERROR) << "模型初始化失败: " << init_status.get_err_msg();
        return false;
      }
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "初始化异常: " << e.what();
      return false;
    }
  }

  // 格式化聊天历史为ChatML格式
  std::string format_messages(const std::vector<ChatMessage>& messages) const {
    std::string prompt;

    // 按照ChatML格式格式化每条消息
    for (const auto& message : messages) {
      prompt += "<|im_start|>" + message.role + "\n";
      prompt += message.content + "\n";
      prompt += "<|im_end|>\n";
    }

    // 添加助手回复的开头部分
    prompt += "<|im_start|>assistant\n";

    return prompt;
  }

  // 生成文本回复
  std::string generate(const std::string& prompt, int max_length) {
    auto tokens = model_->encode(prompt);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "输入tokens为空。";

    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    bool is_prompt = true;
    const auto& prompt_embedding = model_->embedding(tokens);
    tensor::Tensor pos_tensor = model_->get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> words;
    words.push_back(next);

    while (pos < max_length) {
      pos_tensor.index<int32_t>(0) = pos;

      if (pos < prompt_len - 1) {
        tensor::Tensor input = model_->fill_input(pos_tensor, prompt_embedding, is_prompt);
        model_->predict(input, pos_tensor, is_prompt, next);
      } else {
        is_prompt = false;
        std::vector<int32_t> current_tokens = {next};
        const auto& token_embedding = model_->embedding(current_tokens);
        tensor::Tensor input = model_->fill_input(pos_tensor, token_embedding, is_prompt);
        model_->predict(input, pos_tensor, is_prompt, next);
      }

      if (is_prompt) {
        next = tokens.at(pos + 1);
        words.push_back(next);
      } else {
        words.push_back(next);
        // 检查是否生成了结束标记
        if (pos >= 3) {
          auto decoded = model_->decode(std::vector<int32_t>(words.end() - 4, words.end()));
          if (decoded.find("<|im_end|>") != std::string::npos ||
              decoded.find("<|endoftext|>") != std::string::npos) {
            break;
          }
        }
      }

      pos += 1;
    }

    // 提取生成的回复，去除提示部分
    std::vector<int32_t> response_tokens(words.begin() + prompt_len, words.end());
    std::string response = model_->decode(response_tokens);

    // 移除结束标记
    size_t end_pos = response.find("<|im_end|>");
    if (end_pos != std::string::npos) {
      response = response.substr(0, end_pos);
    }

    return response;
  }

  // 聊天主方法
  ChatMessage chat(const std::vector<ChatMessage>& messages, const GenerationConfig& config) {
    // 格式化消息
    std::string prompt = format_messages(messages);

    // 生成回复
    std::string response_text = generate(prompt, config.max_length);

    // 创建并返回回复消息
    ChatMessage response;
    response.role = "assistant";
    response.content = response_text;

    return response;
  }

 private:
  std::string model_path_;
  std::string tokenizer_path_;
  std::unique_ptr<model::Qwen2Model> model_;
};

int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "用法: ./chat <模型路径> <分词器路径>";
    return -1;
  }

  google::InitGoogleLogging(argv[0]);

  const char* model_path = argv[1];
  const char* tokenizer_path = argv[2];

  // 创建聊天助手实例
  ChatAssistant assistant(model_path, tokenizer_path);

  // 初始化助手
  if (!assistant.init()) {
    LOG(FATAL) << "聊天助手初始化失败!";
    return -1;
  }

  // 初始化聊天历史
  std::vector<ChatMessage> chat_history;

  // 添加系统提示
  chat_history.push_back({"system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."});

  // 设置生成参数
  GenerationConfig gen_config;

  std::string user_input;
  bool first_message = true;

  std::cout << "初始化完成。输入'退出'结束聊天。\n" << std::endl;

  // 打印历史信息的函数
  auto print_history = [](const std::vector<ChatMessage>& history) {
    std::cout << "\n=== 当前对话历史 ===" << std::endl;
    for (const auto& msg : history) {
      std::cout << msg.role << ": " << msg.content << std::endl;
    }
    std::cout << "==================\n" << std::endl;
  };

  // 打印ChatML格式prompt的函数
  auto print_chatml_prompt = [&assistant](const std::vector<ChatMessage>& history) {
    std::cout << "\n=== ChatML格式Prompt ===" << std::endl;
    std::string prompt = assistant.format_messages(history);
    std::cout << prompt;
    std::cout << "==================\n" << std::endl;
  };

  while (true) {
    if (first_message) {
      std::cout << "请输入问题: ";
      first_message = false;
    } else {
      std::cout << "\n请输入问题: ";
    }

    // 获取用户输入
    std::getline(std::cin, user_input);

    // 检查是否退出
    if (user_input == "quit") {
      break;
    }

    // 添加用户消息到历史
    chat_history.push_back({"user", user_input});

    // 生成并显示模型回复
    std::cout << "\n助手: " << std::flush;

    auto start = std::chrono::steady_clock::now();

    // 生成回复
    ChatMessage response = assistant.chat(chat_history, gen_config);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    std::cout << response.content << std::endl;
    std::cout << "\n[生成时间: " << duration << "秒]" << std::endl;

    // 将助手回复添加到历史
    chat_history.push_back(response);

    // 打印当前对话历史和ChatML格式prompt
    // print_history(chat_history);
    // print_chatml_prompt(chat_history);
  }

  std::cout << "聊天已结束。" << std::endl;
  return 0;
}
