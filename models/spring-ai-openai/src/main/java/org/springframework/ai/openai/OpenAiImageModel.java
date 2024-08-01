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
import org.springframework.ai.image.Image;
import org.springframework.ai.image.ImageGeneration;
import org.springframework.ai.image.ImageModel;
import org.springframework.ai.image.ImageOptions;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.ai.image.ImageResponse;
import org.springframework.ai.image.ImageResponseMetadata;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiImageApi;
import org.springframework.ai.openai.metadata.OpenAiImageGenerationMetadata;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;

import java.util.List;

/**
 * OpenAiImageModel is a class that implements the ImageModel interface. It provides a
 * client for calling the OpenAI image generation API.
 *
 * @author Mark Pollack
 * @author Christian Tzolov
 * @author Hyunjoon Choi
 * @author Thomas Vitale
 * @since 0.8.0
 */
public class OpenAiImageModel implements ImageModel {

	private final static Logger logger = LoggerFactory.getLogger(OpenAiImageModel.class);

	/**
	 * The default options used for the image completion requests.
	 */
	private final OpenAiImageOptions defaultOptions;

	/**
	 * The retry template used to retry the OpenAI Image API calls.
	 */
	private final RetryTemplate retryTemplate;

	/**
	 * Low-level access to the OpenAI Image API.
	 */
	private final OpenAiImageApi openAiImageApi;

	/**
	 * Creates an instance of the OpenAiImageModel.
	 * @param openAiImageApi The OpenAiImageApi instance to be used for interacting with
	 * the OpenAI Image API.
	 * @throws IllegalArgumentException if openAiImageApi is null
	 */
	public OpenAiImageModel(OpenAiImageApi openAiImageApi) {
		this(openAiImageApi, OpenAiImageOptions.builder().build(), RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the OpenAiImageModel.
	 * @param openAiImageApi The OpenAiImageApi instance to be used for interacting with
	 * the OpenAI Image API.
	 * @param options The OpenAiImageOptions to configure the image model.
	 * @param retryTemplate The retry template.
	 */
	public OpenAiImageModel(OpenAiImageApi openAiImageApi, OpenAiImageOptions options, RetryTemplate retryTemplate) {
		Assert.notNull(openAiImageApi, "OpenAiImageApi must not be null");
		Assert.notNull(options, "options must not be null");
		Assert.notNull(retryTemplate, "retryTemplate must not be null");
		this.openAiImageApi = openAiImageApi;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
	}

	@Override
	public ImageResponse call(ImagePrompt imagePrompt) {
		OpenAiImageOptions requestImageOptions = mergeOptions(imagePrompt.getOptions(), this.defaultOptions);
		OpenAiImageApi.OpenAiImageRequest imageRequest = createRequest(imagePrompt, requestImageOptions);

		ResponseEntity<OpenAiImageApi.OpenAiImageResponse> imageResponseEntity = this.retryTemplate
			.execute(ctx -> this.openAiImageApi.createImage(imageRequest));

		return convertResponse(imageResponseEntity, imageRequest);
	}

	private OpenAiImageApi.OpenAiImageRequest createRequest(ImagePrompt imagePrompt,
			OpenAiImageOptions mergedImageOptions) {
		String instructions = imagePrompt.getInstructions().get(0).getText();

		OpenAiImageApi.OpenAiImageRequest imageRequest = new OpenAiImageApi.OpenAiImageRequest(instructions,
				OpenAiImageApi.DEFAULT_IMAGE_MODEL);

		return ModelOptionsUtils.merge(mergedImageOptions, imageRequest, OpenAiImageApi.OpenAiImageRequest.class);
	}

	private ImageResponse convertResponse(ResponseEntity<OpenAiImageApi.OpenAiImageResponse> imageResponseEntity,
			OpenAiImageApi.OpenAiImageRequest openAiImageRequest) {
		OpenAiImageApi.OpenAiImageResponse imageApiResponse = imageResponseEntity.getBody();
		if (imageApiResponse == null) {
			logger.warn("No image response returned for request: {}", openAiImageRequest);
			return new ImageResponse(List.of());
		}

		List<ImageGeneration> imageGenerationList = imageApiResponse.data()
			.stream()
			.map(entry -> new ImageGeneration(new Image(entry.url(), entry.b64Json()),
					new OpenAiImageGenerationMetadata(entry.revisedPrompt())))
			.toList();

		ImageResponseMetadata openAiImageResponseMetadata = new ImageResponseMetadata(imageApiResponse.created());
		return new ImageResponse(imageGenerationList, openAiImageResponseMetadata);
	}

	/**
	 * Merge runtime and default {@link ImageOptions} to compute the final options to use
	 * in the request.
	 */
	private OpenAiImageOptions mergeOptions(ImageOptions runtimeOptions, OpenAiImageOptions defaultOptions) {
		if (runtimeOptions == null) {
			return defaultOptions;
		}

		return OpenAiImageOptions.builder()
			// Handle portable image options
			.withModel(ModelOptionsUtils.mergeOption(runtimeOptions.getModel(), defaultOptions.getModel()))
			.withN(ModelOptionsUtils.mergeOption(runtimeOptions.getN(), defaultOptions.getN()))
			.withResponseFormat(ModelOptionsUtils.mergeOption(runtimeOptions.getResponseFormat(),
					defaultOptions.getResponseFormat()))
			.withWidth(ModelOptionsUtils.mergeOption(runtimeOptions.getWidth(), defaultOptions.getWidth()))
			.withHeight(ModelOptionsUtils.mergeOption(runtimeOptions.getHeight(), defaultOptions.getHeight()))
			.withStyle(ModelOptionsUtils.mergeOption(runtimeOptions.getStyle(), defaultOptions.getStyle()))
			// Handle OpenAI specific image options
			.withQuality(defaultOptions.getQuality())
			.withUser(defaultOptions.getUser())
			.build();
	}

}
