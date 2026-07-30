#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen3.h"
int32_t generate(const model::Qwen3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (next != 151645 && next != 151644) {
        words.push_back(next);
      }
    }
    if (model.is_sentence_ending(next)) {
      break;
    }

    if (is_prompt) {
      next = tokens.at(pos + 1);
    }
    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

std::string fill_template(const std::string& content) {
  const std::string format =
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
  std::string result = format;
  size_t pos = result.find("%s");
  if (pos != std::string::npos) {
    result.replace(pos, 2, content);
  }
  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }

  std::string hi = "What is AI?";
  std::cout << hi << "\n";
  const std::string& sentence = fill_template(hi);
  auto start = std::chrono::steady_clock::now();
  fflush(stdout);
  int steps = generate(model, sentence, 2560, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps:%d\n", steps);
  printf("\nduration:%lf\n", duration);
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}