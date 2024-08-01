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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiApi.EmbeddingList;
import org.springframework.ai.openai.metadata.OpenAiUsage;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.lang.Nullable;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;

import java.util.List;

/**
 * Open AI Embedding Model implementation.
 *
 * @author Christian Tzolov
 * @author Thomas Vitale
 */
public class OpenAiEmbeddingModel extends AbstractEmbeddingModel {

	private static final Logger logger = LoggerFactory.getLogger(OpenAiEmbeddingModel.class);

	private final OpenAiEmbeddingOptions defaultOptions;

	private final RetryTemplate retryTemplate;

	private final OpenAiApi openAiApi;

	private final MetadataMode metadataMode;

	/**
	 * Constructor for the OpenAiEmbeddingModel class.
	 * @param openAiApi The OpenAiApi instance to use for making API requests.
	 */
	public OpenAiEmbeddingModel(OpenAiApi openAiApi) {
		this(openAiApi, MetadataMode.EMBED);
	}

	/**
	 * Initializes a new instance of the OpenAiEmbeddingModel class.
	 * @param openAiApi The OpenAiApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 */
	public OpenAiEmbeddingModel(OpenAiApi openAiApi, MetadataMode metadataMode) {
		this(openAiApi, metadataMode,
				OpenAiEmbeddingOptions.builder().withModel(OpenAiApi.DEFAULT_EMBEDDING_MODEL).build());
	}

	/**
	 * Initializes a new instance of the OpenAiEmbeddingModel class.
	 * @param openAiApi The OpenAiApi instance to use for making API requests.
	 * @param metadataMode The mode for generating metadata.
	 * @param openAiEmbeddingOptions The options for OpenAi embedding.
	 */
	public OpenAiEmbeddingModel(OpenAiApi openAiApi, MetadataMode metadataMode,
			OpenAiEmbeddingOptions openAiEmbeddingOptions) {
		this(openAiApi, metadataMode, openAiEmbeddingOptions, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the OpenAiEmbeddingModel class.
	 * @param openAiApi - The OpenAiApi instance to use for making API requests.
	 * @param metadataMode - The mode for generating metadata.
	 * @param options - The options for OpenAI embedding.
	 * @param retryTemplate - The RetryTemplate for retrying failed API requests.
	 */
	public OpenAiEmbeddingModel(OpenAiApi openAiApi, MetadataMode metadataMode, OpenAiEmbeddingOptions options,
			RetryTemplate retryTemplate) {
		Assert.notNull(openAiApi, "openAiApi must not be null");
		Assert.notNull(metadataMode, "metadataMode must not be null");
		Assert.notNull(options, "options must not be null");
		Assert.notNull(retryTemplate, "retryTemplate must not be null");

		this.openAiApi = openAiApi;
		this.metadataMode = metadataMode;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
	}

	@Override
	public List<Double> embed(Document document) {
		Assert.notNull(document, "Document must not be null");
		return this.embed(document.getFormattedContent(this.metadataMode));
	}

	@Override
	public EmbeddingResponse call(EmbeddingRequest request) {
		OpenAiEmbeddingOptions requestOptions = mergeOptions(request.getOptions(), this.defaultOptions);

		OpenAiApi.EmbeddingRequest<List<String>> apiRequest = createRequest(request, requestOptions);

		EmbeddingList<OpenAiApi.Embedding> apiEmbeddingResponse = this.retryTemplate
			.execute(ctx -> this.openAiApi.embeddings(apiRequest).getBody());

		if (apiEmbeddingResponse == null) {
			logger.warn("No embeddings returned for request: {}", request);
			return new EmbeddingResponse(List.of());
		}

		var metadata = new EmbeddingResponseMetadata(apiEmbeddingResponse.model(),
				OpenAiUsage.from(apiEmbeddingResponse.usage()));

		List<Embedding> embeddings = apiEmbeddingResponse.data()
			.stream()
			.map(e -> new Embedding(e.embedding(), e.index()))
			.toList();

		return new EmbeddingResponse(embeddings, metadata);
	}

	private OpenAiApi.EmbeddingRequest<List<String>> createRequest(EmbeddingRequest request,
			OpenAiEmbeddingOptions requestOptions) {
		return new OpenAiApi.EmbeddingRequest<>(request.getInstructions(), requestOptions.getModel(),
				requestOptions.getEncodingFormat(), requestOptions.getDimensions(), requestOptions.getUser());
	}

	/**
	 * Merge runtime and default {@link EmbeddingOptions} to compute the final options to
	 * use in the request.
	 */
	private OpenAiEmbeddingOptions mergeOptions(@Nullable EmbeddingOptions runtimeOptions,
			OpenAiEmbeddingOptions defaultOptions) {
		if (runtimeOptions == null) {
			return defaultOptions;
		}

		return OpenAiEmbeddingOptions.builder()
			// Handle portable embedding options
			.withModel(ModelOptionsUtils.mergeOption(runtimeOptions.getModel(), defaultOptions.getModel()))
			.withDimensions(
					ModelOptionsUtils.mergeOption(runtimeOptions.getDimensions(), defaultOptions.getDimensions()))
			// Handle OpenAI specific embedding options
			.withEncodingFormat(defaultOptions.getEncodingFormat())
			.withUser(defaultOptions.getUser())
			.build();
	}

}
