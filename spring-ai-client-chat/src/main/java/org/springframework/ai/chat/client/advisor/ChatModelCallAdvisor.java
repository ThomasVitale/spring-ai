/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.chat.client.advisor;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisorChain;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;

/**
 * A {@link CallAdvisor} that uses a {@link ChatModel} to generate a response.
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class ChatModelCallAdvisor implements CallAdvisor {

	private final ChatModel chatModel;

	public ChatModelCallAdvisor(ChatModel chatModel) {
		this.chatModel = chatModel;
	}

	@Override
	public ChatClientResponse adviseCall(ChatClientRequest chatClientRequest, CallAroundAdvisorChain chain) {
		Assert.notNull(chatClientRequest, "the chatClientRequest cannot be null");

		ChatMemory chatMemory = chain.getChatMemory();

		ChatResponse chatResponse;
		if (chatMemory == null) {
			chatResponse = chatModel.call(chatClientRequest.prompt());
		}
		else {
			String conversationId = chain.getConversationId();
			chatMemory.add(conversationId, chatClientRequest.prompt().getInstructions());
			Prompt prompt = chatClientRequest.prompt().mutate().messages(chatMemory.get(conversationId)).build();
			chatResponse = chatModel.call(prompt);
			if (chatResponse != null) {
				List<Generation> generations = chatResponse.getResults();
				if (generations != null) {
					List<Message> assistantMessages = generations.stream()
						.map(generation -> (Message) generation.getOutput())
						.toList();
					chatMemory.add(conversationId, assistantMessages);
				}
			}

		}

		return ChatClientResponse.builder()
			.chatResponse(chatResponse)
			.context(Map.copyOf(chatClientRequest.context()))
			.build();
	}

	@Override
	public String getName() {
		return "call";
	}

	@Override
	public int getOrder() {
		return Ordered.LOWEST_PRECEDENCE;
	}

}
