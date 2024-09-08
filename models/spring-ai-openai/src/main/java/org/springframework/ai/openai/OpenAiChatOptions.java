/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.openai;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallingOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest.ResponseFormat;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest.StreamOptions;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest.ToolChoiceBuilder;
import org.springframework.ai.openai.api.OpenAiApi.FunctionTool;
import org.springframework.boot.context.properties.NestedConfigurationProperty;
import org.springframework.util.Assert;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * @author Christian Tzolov
 * @author Mariusz Bernacki
 * @author Thomas Vitale
 * @since 0.8.0
 */
@JsonInclude(Include.NON_NULL)
public class OpenAiChatOptions implements FunctionCallingOptions, ChatOptions {

	// @formatter:off
	/**
	 * ID of the model to use.
	 */
	private @JsonProperty("model") String model;
	/**
	 * Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
	 * frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
	 */
	private @JsonProperty("frequency_penalty") Double frequencyPenalty;
	/**
	 * Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object
	 * that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
	 * Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will
	 * vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100
	 * or 100 should result in a ban or exclusive selection of the relevant token.
	 */
	private @JsonProperty("logit_bias") Map<String, Integer> logitBias;
	/**
	 * Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities
	 * of each output token returned in the 'content' of 'message'.
	 */
	private @JsonProperty("logprobs") Boolean logprobs;
	/**
	 * An integer between 0 and 5 specifying the number of most likely tokens to return at each token position,
	 * each with an associated log probability. 'logprobs' must be set to 'true' if this parameter is used.
	 */
	private @JsonProperty("top_logprobs") Integer topLogprobs;
	/**
	 * The maximum number of tokens to generate in the chat completion. The total length of input
	 * tokens and generated tokens is limited by the model's context length.
	 */
	private @JsonProperty("max_tokens") Integer maxTokens;
	/**
	 * How many chat completion choices to generate for each input message. Note that you will be charged based
	 * on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
	 */
	private @JsonProperty("n") Integer n;
	/**
	 * Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
	 * appear in the text so far, increasing the model's likelihood to talk about new topics.
	 */
	private @JsonProperty("presence_penalty") Double presencePenalty;
	/**
	 * An object specifying the format that the model must output. Setting to { "type":
	 * "json_object" } enables JSON mode, which guarantees the message the model generates is valid JSON.
	 */
	private @JsonProperty("response_format") ResponseFormat responseFormat;
	/**
	 * Options for streaming response. Included in the API only if streaming-mode completion is requested.
	 */
	private @JsonProperty("stream_options") StreamOptions streamOptions;
	/**
	 * This feature is in Beta. If specified, our system will make a best effort to sample
	 * deterministically, such that repeated requests with the same seed and parameters should return the same result.
	 * Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor
	 * changes in the backend.
	 */
	private @JsonProperty("seed") Integer seed;
	/**
	 * Up to 4 sequences where the API will stop generating further tokens.
	 */
	@NestedConfigurationProperty
	private @JsonProperty("stop") List<String> stop;
	/**
	 * What sampling temperature to use, between 0 and 1. Higher values like 0.8 will make the output
	 * more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend
	 * altering this or top_p but not both.
	 */
	private @JsonProperty("temperature") Double temperature;
	/**
	 * An alternative to sampling with temperature, called nucleus sampling, where the model considers the
	 * results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10%
	 * probability mass are considered. We generally recommend altering this or temperature but not both.
	 */
	private @JsonProperty("top_p") Double topP;
	/**
	 * A list of tools the model may call. Currently, only functions are supported as a tool. Use this to
	 * provide a list of functions the model may generate JSON inputs for.
	 */
	@NestedConfigurationProperty
	private @JsonProperty("tools") List<FunctionTool> tools;
	/**
	 * Controls which (if any) function is called by the model. none means the model will not call a
	 * function and instead generates a message. auto means the model can pick between generating a message or calling a
	 * function. Specifying a particular function via {"type: "function", "function": {"name": "my_function"}} forces
	 * the model to call that function. none is the default when no functions are present. auto is the default if
	 * functions are present. Use the {@link ToolChoiceBuilder} to create a tool choice object.
	 */
	private @JsonProperty("tool_choice") String toolChoice;
	/**
	 * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
	 */
	private @JsonProperty("user") String user;
	/**
	 * Whether to enable <a href="https://platform.openai.com/docs/guides/function-calling/parallel-function-calling">parallel function calling</a> during tool use.
	 * Defaults to true.
	 */
	private @JsonProperty("parallel_tool_calls") Boolean parallelToolCalls;

	/**
	 * OpenAI Tool Function Callbacks to register with the ChatModel.
	 * For Prompt Options the functionCallbacks are automatically enabled for the duration of the prompt execution.
	 * For Default Options the functionCallbacks are registered but disabled by default. Use the enableFunctions to set the functions
	 * from the registry to be used by the ChatModel chat completion requests.
	 */
	@NestedConfigurationProperty
	@JsonIgnore
	private List<FunctionCallback> functionCallbacks = new ArrayList<>();

	/**
	 * List of functions, identified by their names, to configure for function calling in
	 * the chat completion requests.
	 * Functions with those names must exist in the functionCallbacks registry.
	 * The {@link #functionCallbacks} from the PromptOptions are automatically enabled for the duration of the prompt execution.
	 *
	 * Note that function enabled with the default options are enabled for all chat completion requests. This could impact the token count and the billing.
	 * If the functions is set in a prompt options, then the enabled functions are only active for the duration of this prompt execution.
	 */
	@NestedConfigurationProperty
	@JsonIgnore
	private Set<String> functions = new HashSet<>();

	/**
	 * Optional HTTP headers to be added to the chat completion request.
	 */
	@NestedConfigurationProperty
	@JsonIgnore
	private Map<String, String> httpHeaders = new HashMap<>();
	// @formatter:on

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {

		protected OpenAiChatOptions options;

		public Builder() {
			this.options = new OpenAiChatOptions();
		}

		public Builder(OpenAiChatOptions options) {
			this.options = options;
		}

		public Builder withModel(String model) {
			this.options.model = model;
			return this;
		}

		public Builder withModel(OpenAiApi.ChatModel openAiChatModel) {
			this.options.model = openAiChatModel.getName();
			return this;
		}

		public Builder withFrequencyPenalty(Double frequencyPenalty) {
			this.options.frequencyPenalty = frequencyPenalty;
			return this;
		}

		public Builder withLogitBias(Map<String, Integer> logitBias) {
			this.options.logitBias = logitBias;
			return this;
		}

		public Builder withLogprobs(Boolean logprobs) {
			this.options.logprobs = logprobs;
			return this;
		}

		public Builder withTopLogprobs(Integer topLogprobs) {
			this.options.topLogprobs = topLogprobs;
			return this;
		}

		public Builder withMaxTokens(Integer maxTokens) {
			this.options.maxTokens = maxTokens;
			return this;
		}

		public Builder withN(Integer n) {
			this.options.n = n;
			return this;
		}

		public Builder withPresencePenalty(Double presencePenalty) {
			this.options.presencePenalty = presencePenalty;
			return this;
		}

		public Builder withResponseFormat(ResponseFormat responseFormat) {
			this.options.responseFormat = responseFormat;
			return this;
		}

		public Builder withStreamUsage(boolean enableStreamUsage) {
			this.options.streamOptions = (enableStreamUsage) ? StreamOptions.INCLUDE_USAGE : null;
			return this;
		}

		public Builder withSeed(Integer seed) {
			this.options.seed = seed;
			return this;
		}

		public Builder withStop(List<String> stop) {
			this.options.stop = stop;
			return this;
		}

		public Builder withTemperature(Double temperature) {
			this.options.temperature = temperature;
			return this;
		}

		public Builder withTopP(Double topP) {
			this.options.topP = topP;
			return this;
		}

		public Builder withTools(List<FunctionTool> tools) {
			this.options.tools = tools;
			return this;
		}

		public Builder withToolChoice(String toolChoice) {
			this.options.toolChoice = toolChoice;
			return this;
		}

		public Builder withUser(String user) {
			this.options.user = user;
			return this;
		}

		public Builder withParallelToolCalls(Boolean parallelToolCalls) {
			this.options.parallelToolCalls = parallelToolCalls;
			return this;
		}

		public Builder withFunctionCallbacks(List<FunctionCallback> functionCallbacks) {
			this.options.functionCallbacks = functionCallbacks;
			return this;
		}

		public Builder withFunctions(Set<String> functionNames) {
			Assert.notNull(functionNames, "Function names must not be null");
			this.options.functions = functionNames;
			return this;
		}

		public Builder withFunction(String functionName) {
			Assert.hasText(functionName, "Function name must not be empty");
			this.options.functions.add(functionName);
			return this;
		}

		public Builder withHttpHeaders(Map<String, String> httpHeaders) {
			Assert.notNull(httpHeaders, "HTTP headers must not be null");
			this.options.httpHeaders = httpHeaders;
			return this;
		}

		public OpenAiChatOptions build() {
			return this.options;
		}

	}

	public Boolean getStreamUsage() {
		return this.streamOptions != null;
	}

	public void setStreamUsage(Boolean enableStreamUsage) {
		this.streamOptions = (enableStreamUsage) ? StreamOptions.INCLUDE_USAGE : null;
	}

	@Override
	public String getModel() {
		return this.model;
	}

	public void setModel(String model) {
		this.model = model;
	}

	@Override
	public Double getFrequencyPenalty() {
		return this.frequencyPenalty;
	}

	public void setFrequencyPenalty(Double frequencyPenalty) {
		this.frequencyPenalty = frequencyPenalty;
	}

	public Map<String, Integer> getLogitBias() {
		return this.logitBias;
	}

	public void setLogitBias(Map<String, Integer> logitBias) {
		this.logitBias = logitBias;
	}

	public Boolean getLogprobs() {
		return this.logprobs;
	}

	public void setLogprobs(Boolean logprobs) {
		this.logprobs = logprobs;
	}

	public Integer getTopLogprobs() {
		return this.topLogprobs;
	}

	public void setTopLogprobs(Integer topLogprobs) {
		this.topLogprobs = topLogprobs;
	}

	@Override
	public Integer getMaxTokens() {
		return this.maxTokens;
	}

	public void setMaxTokens(Integer maxTokens) {
		this.maxTokens = maxTokens;
	}

	public Integer getN() {
		return this.n;
	}

	public void setN(Integer n) {
		this.n = n;
	}

	@Override
	public Double getPresencePenalty() {
		return this.presencePenalty;
	}

	public void setPresencePenalty(Double presencePenalty) {
		this.presencePenalty = presencePenalty;
	}

	public ResponseFormat getResponseFormat() {
		return this.responseFormat;
	}

	public void setResponseFormat(ResponseFormat responseFormat) {
		this.responseFormat = responseFormat;
	}

	public StreamOptions getStreamOptions() {
		return streamOptions;
	}

	public void setStreamOptions(StreamOptions streamOptions) {
		this.streamOptions = streamOptions;
	}

	public Integer getSeed() {
		return this.seed;
	}

	public void setSeed(Integer seed) {
		this.seed = seed;
	}

	@Override
	@JsonIgnore
	public List<String> getStopSequences() {
		return getStop();
	}

	@JsonIgnore
	public void setStopSequences(List<String> stopSequences) {
		setStop(stopSequences);
	}

	public List<String> getStop() {
		return this.stop;
	}

	public void setStop(List<String> stop) {
		this.stop = stop;
	}

	@Override
	public Double getTemperature() {
		return this.temperature;
	}

	public void setTemperature(Double temperature) {
		this.temperature = temperature;
	}

	@Override
	public Double getTopP() {
		return this.topP;
	}

	public void setTopP(Double topP) {
		this.topP = topP;
	}

	public List<FunctionTool> getTools() {
		return this.tools;
	}

	public void setTools(List<FunctionTool> tools) {
		this.tools = tools;
	}

	public String getToolChoice() {
		return this.toolChoice;
	}

	public void setToolChoice(String toolChoice) {
		this.toolChoice = toolChoice;
	}

	public String getUser() {
		return this.user;
	}

	public void setUser(String user) {
		this.user = user;
	}

	public Boolean getParallelToolCalls() {
		return this.parallelToolCalls;
	}

	public void setParallelToolCalls(Boolean parallelToolCalls) {
		this.parallelToolCalls = parallelToolCalls;
	}

	@Override
	public List<FunctionCallback> getFunctionCallbacks() {
		return this.functionCallbacks;
	}

	@Override
	public void setFunctionCallbacks(List<FunctionCallback> functionCallbacks) {
		this.functionCallbacks = functionCallbacks;
	}

	@Override
	public Set<String> getFunctions() {
		return functions;
	}

	public void setFunctions(Set<String> functionNames) {
		this.functions = functionNames;
	}

	public Map<String, String> getHttpHeaders() {
		return this.httpHeaders;
	}

	public void setHttpHeaders(Map<String, String> httpHeaders) {
		this.httpHeaders = httpHeaders;
	}

	@Override
	@JsonIgnore
	public Integer getTopK() {
		return null;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((model == null) ? 0 : model.hashCode());
		result = prime * result + ((frequencyPenalty == null) ? 0 : frequencyPenalty.hashCode());
		result = prime * result + ((logitBias == null) ? 0 : logitBias.hashCode());
		result = prime * result + ((logprobs == null) ? 0 : logprobs.hashCode());
		result = prime * result + ((topLogprobs == null) ? 0 : topLogprobs.hashCode());
		result = prime * result + ((maxTokens == null) ? 0 : maxTokens.hashCode());
		result = prime * result + ((n == null) ? 0 : n.hashCode());
		result = prime * result + ((presencePenalty == null) ? 0 : presencePenalty.hashCode());
		result = prime * result + ((responseFormat == null) ? 0 : responseFormat.hashCode());
		result = prime * result + ((streamOptions == null) ? 0 : streamOptions.hashCode());
		result = prime * result + ((seed == null) ? 0 : seed.hashCode());
		result = prime * result + ((stop == null) ? 0 : stop.hashCode());
		result = prime * result + ((temperature == null) ? 0 : temperature.hashCode());
		result = prime * result + ((topP == null) ? 0 : topP.hashCode());
		result = prime * result + ((tools == null) ? 0 : tools.hashCode());
		result = prime * result + ((toolChoice == null) ? 0 : toolChoice.hashCode());
		result = prime * result + ((user == null) ? 0 : user.hashCode());
		result = prime * result + ((parallelToolCalls == null) ? 0 : parallelToolCalls.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		OpenAiChatOptions other = (OpenAiChatOptions) obj;
		if (this.model == null) {
			if (other.model != null)
				return false;
		}
		else if (!model.equals(other.model))
			return false;
		if (this.frequencyPenalty == null) {
			if (other.frequencyPenalty != null)
				return false;
		}
		else if (!this.frequencyPenalty.equals(other.frequencyPenalty))
			return false;
		if (this.logitBias == null) {
			if (other.logitBias != null)
				return false;
		}
		else if (!this.logitBias.equals(other.logitBias))
			return false;
		if (this.logprobs == null) {
			if (other.logprobs != null)
				return false;
		}
		else if (!this.logprobs.equals(other.logprobs))
			return false;
		if (this.topLogprobs == null) {
			if (other.topLogprobs != null)
				return false;
		}
		else if (!this.topLogprobs.equals(other.topLogprobs))
			return false;
		if (this.maxTokens == null) {
			if (other.maxTokens != null)
				return false;
		}
		else if (!this.maxTokens.equals(other.maxTokens))
			return false;
		if (this.n == null) {
			if (other.n != null)
				return false;
		}
		else if (!this.n.equals(other.n))
			return false;
		if (this.presencePenalty == null) {
			if (other.presencePenalty != null)
				return false;
		}
		else if (!this.presencePenalty.equals(other.presencePenalty))
			return false;
		if (this.responseFormat == null) {
			if (other.responseFormat != null)
				return false;
		}
		else if (!this.responseFormat.equals(other.responseFormat))
			return false;
		if (this.streamOptions == null) {
			if (other.streamOptions != null)
				return false;
		}
		else if (!this.streamOptions.equals(other.streamOptions))
			return false;
		if (this.seed == null) {
			if (other.seed != null)
				return false;
		}
		else if (!this.seed.equals(other.seed))
			return false;
		if (this.stop == null) {
			if (other.stop != null)
				return false;
		}
		else if (!stop.equals(other.stop))
			return false;
		if (this.temperature == null) {
			if (other.temperature != null)
				return false;
		}
		else if (!this.temperature.equals(other.temperature))
			return false;
		if (this.topP == null) {
			if (other.topP != null)
				return false;
		}
		else if (!topP.equals(other.topP))
			return false;
		if (this.tools == null) {
			if (other.tools != null)
				return false;
		}
		else if (!tools.equals(other.tools))
			return false;
		if (this.toolChoice == null) {
			if (other.toolChoice != null)
				return false;
		}
		else if (!toolChoice.equals(other.toolChoice))
			return false;
		if (this.user == null) {
			if (other.user != null)
				return false;
		}
		else if (!this.user.equals(other.user))
			return false;
		else if (this.parallelToolCalls == null) {
			if (other.parallelToolCalls != null)
				return false;
		}
		else if (!this.parallelToolCalls.equals(other.parallelToolCalls))
			return false;

		return true;
	}

	@Override
	public OpenAiChatOptions copy() {
		return OpenAiChatOptions.fromOptions(this);
	}

	public static OpenAiChatOptions fromOptions(OpenAiChatOptions fromOptions) {
		return OpenAiChatOptions.builder()
			.withModel(fromOptions.getModel())
			.withFrequencyPenalty(fromOptions.getFrequencyPenalty())
			.withLogitBias(fromOptions.getLogitBias())
			.withLogprobs(fromOptions.getLogprobs())
			.withTopLogprobs(fromOptions.getTopLogprobs())
			.withMaxTokens(fromOptions.getMaxTokens())
			.withN(fromOptions.getN())
			.withPresencePenalty(fromOptions.getPresencePenalty())
			.withResponseFormat(fromOptions.getResponseFormat())
			.withStreamUsage(fromOptions.getStreamUsage())
			.withSeed(fromOptions.getSeed())
			.withStop(fromOptions.getStop())
			.withTemperature(fromOptions.getTemperature())
			.withTopP(fromOptions.getTopP())
			.withTools(fromOptions.getTools())
			.withToolChoice(fromOptions.getToolChoice())
			.withUser(fromOptions.getUser())
			.withParallelToolCalls(fromOptions.getParallelToolCalls())
			.withFunctionCallbacks(fromOptions.getFunctionCallbacks())
			.withFunctions(fromOptions.getFunctions())
			.withHttpHeaders(fromOptions.getHttpHeaders())
			.build();
	}

	@Override
	public String toString() {
		return "OpenAiChatOptions: " + ModelOptionsUtils.toJsonString(this);
	}

}
