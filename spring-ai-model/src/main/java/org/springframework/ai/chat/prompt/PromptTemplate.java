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

package org.springframework.ai.chat.prompt;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.template.TemplateRenderer;
import org.springframework.ai.chat.prompt.template.st.StTemplateRenderer;
import org.springframework.ai.content.Media;
import org.springframework.core.io.Resource;
import org.springframework.util.Assert;
import org.springframework.util.StreamUtils;

/**
 * A template for creating prompts. It allows you to define a template string with
 * placeholders for variables, and then render the template with specific values for those
 * variables.
 *
 * NOTE: This class will be marked as final in the next release. If you subclass this
 * class, you should consider using the built-in implementation together with the new
 * PromptTemplateRenderer interface, which is designed to give you more flexibility and
 * control over the rendering process.
 */
public class PromptTemplate implements PromptTemplateActions, PromptTemplateMessageActions {

	private static final TemplateRenderer DEFAULT_TEMPLATE_RENDERER = StTemplateRenderer.builder().build();

	/**
	 * @deprecated will become private in the next release. If you're subclassing this
	 * class, re-consider using the built-in implementation together with the new
	 * PromptTemplateRenderer interface, designed to give you more flexibility and control
	 * over the rendering process.
	 */
	@Deprecated
	protected String template;

	/**
	 * @deprecated in favor of {@link TemplateRenderer}
	 */
	@Deprecated
	protected TemplateFormat templateFormat = TemplateFormat.ST;

	private final Map<String, Object> variables = new HashMap<>();

	private final TemplateRenderer renderer;

	public PromptTemplate(Resource resource) {
		this(resource, new HashMap<>(), DEFAULT_TEMPLATE_RENDERER);
	}

	public PromptTemplate(String template) {
		this(template, new HashMap<>(), DEFAULT_TEMPLATE_RENDERER);
	}

	public PromptTemplate(String template, Map<String, Object> variables) {
		this(template, variables, DEFAULT_TEMPLATE_RENDERER);
	}

	public PromptTemplate(Resource resource, Map<String, Object> variables) {
		this(resource, variables, DEFAULT_TEMPLATE_RENDERER);
	}

	PromptTemplate(String template, Map<String, Object> variables, TemplateRenderer renderer) {
		Assert.hasText(template, "template cannot be null or empty");
		Assert.notNull(variables, "variables cannot be null");
		Assert.noNullElements(variables.keySet(), "variables keys cannot be null");
		Assert.notNull(renderer, "renderer cannot be null");

		this.template = template;
		this.variables.putAll(variables);
		this.renderer = renderer;
	}

	PromptTemplate(Resource resource, Map<String, Object> variables, TemplateRenderer renderer) {
		Assert.notNull(resource, "resource cannot be null");
		Assert.notNull(variables, "variables cannot be null");
		Assert.noNullElements(variables.keySet(), "variables keys cannot be null");
		Assert.notNull(renderer, "renderer cannot be null");

		try (InputStream inputStream = resource.getInputStream()) {
			this.template = StreamUtils.copyToString(inputStream, Charset.defaultCharset());
			Assert.hasText(template, "template cannot be null or empty");
		}
		catch (IOException ex) {
			throw new RuntimeException("Failed to read resource", ex);
		}
		this.variables.putAll(variables);
		this.renderer = renderer;
	}

	public void add(String name, Object value) {
		this.variables.put(name, value);
	}

	public String getTemplate() {
		return this.template;
	}

	/**
	 * @deprecated in favor of {@link TemplateRenderer}
	 */
	@Deprecated
	public TemplateFormat getTemplateFormat() {
		return this.templateFormat;
	}

	// From PromptTemplateStringActions.

	@Override
	public String render() {
		return this.renderer.apply(template, this.variables);
	}

	@Override
	public String render(Map<String, Object> additionalVariables) {
		Map<String, Object> combinedVariables = new HashMap<>(this.variables);

		for (Entry<String, Object> entry : additionalVariables.entrySet()) {
			if (entry.getValue() instanceof Resource) {
				combinedVariables.put(entry.getKey(), renderResource((Resource) entry.getValue()));
			}
			else {
				combinedVariables.put(entry.getKey(), entry.getValue());
			}
		}

		return this.renderer.apply(template, combinedVariables);
	}

	private String renderResource(Resource resource) {
		try {
			return resource.getContentAsString(Charset.defaultCharset());
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	// From PromptTemplateMessageActions.

	@Override
	public Message createMessage() {
		return new UserMessage(render());
	}

	@Override
	public Message createMessage(List<Media> mediaList) {
		return new UserMessage(render(), mediaList);
	}

	@Override
	public Message createMessage(Map<String, Object> additionalVariables) {
		return new UserMessage(render(additionalVariables));
	}

	// From PromptTemplateActions.

	@Override
	public Prompt create() {
		return new Prompt(render(new HashMap<>()));
	}

	@Override
	public Prompt create(ChatOptions modelOptions) {
		return new Prompt(render(new HashMap<>()), modelOptions);
	}

	@Override
	public Prompt create(Map<String, Object> additionalVariables) {
		return new Prompt(render(additionalVariables));
	}

	@Override
	public Prompt create(Map<String, Object> additionalVariables, ChatOptions modelOptions) {
		return new Prompt(render(additionalVariables), modelOptions);
	}

	// Compatibility

	/**
	 * @deprecated in favor of {@link TemplateRenderer}.
	 */
	@Deprecated
	public Set<String> getInputVariables() {
		throw new UnsupportedOperationException(
				"The template rendering logic is now provided by PromptTemplateRenderer");
	}

	/**
	 * @deprecated in favor of {@link TemplateRenderer}.
	 */
	@Deprecated
	protected void validate(Map<String, Object> model) {
		throw new UnsupportedOperationException("Validation is now provided by the PromptTemplateRenderer");
	}

	public Builder mutate() {
		return new Builder().template(this.template).variables(this.variables).renderer(this.renderer);
	}

	// Builder

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {

		private String template;

		private Resource resource;

		private Map<String, Object> variables = new HashMap<>();

		private TemplateRenderer renderer = DEFAULT_TEMPLATE_RENDERER;

		private Builder() {
		}

		public Builder template(String template) {
			this.template = template;
			return this;
		}

		public Builder resource(Resource resource) {
			this.resource = resource;
			return this;
		}

		public Builder variables(Map<String, Object> variables) {
			this.variables = variables;
			return this;
		}

		public Builder renderer(TemplateRenderer renderer) {
			this.renderer = renderer;
			return this;
		}

		public PromptTemplate build() {
			if (this.template != null && this.resource != null) {
				throw new IllegalArgumentException("Only one of template or resource can be set");
			}
			else if (this.resource != null) {
				return new PromptTemplate(this.resource, this.variables, this.renderer);
			}
			else {
				return new PromptTemplate(this.template, this.variables, this.renderer);
			}
		}

	}

}
