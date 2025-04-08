import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import "dotenv/config";

// Initialize MongoDB client
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

  // Define the MongoDB database and collection
  const dbName = "hr_database";
  const db = client.db(dbName);
  const collection = db.collection("employees");

  /**
   * -----------------------------
   * DEFINE STATE
   * Define the graph state - this just keeps track of messages
   * -----------------------------
   */
  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
    }),
  });


  /**
   * -----------------------------
   * DEFINE TOOLS
   * -----------------------------
   */

  // Define out custom employee lookup tool
  const employeeLookupTool = tool(
    // a query and the number of results to return are passed in
    async ({ query, n = 10 }) => {
      // log the current tool call 
      console.log("Employee lookup tool called");

      // Configure to match the vector index created within mongoDB dashboard
      const dbConfig = {
        collection: collection,
        indexName: "vector_index",
        textKey: "embedding_text",
        embeddingKey: "embedding",
      };

      // Initialize vector store - pass in the embeddings and the database config
      const vectorStore = new MongoDBAtlasVectorSearch(
        new OpenAIEmbeddings(),
        dbConfig
      );

      // Perform the similarity search
      const result = await vectorStore.similaritySearchWithScore(query, n);
      return JSON.stringify(result);
    },
    // Define the tool's metadata
    {
      name: "employee_lookup",
      description: "Gathers employee details from the HR database",
      schema: z.object({
        query: z.string().describe("The search query"),
        n: z
          .number()
          .optional()
          .default(10)
          .describe("Number of results to return"),
      }), // schema defines the args to pass in
    }
  );

  const tools = [employeeLookupTool];
  
  // We can extract the state typing via `GraphState.State`
  // We typehint that this is a possible state
  // A toolnode takes in an array of defined tools
  const toolNode = new ToolNode<typeof GraphState.State>(tools);



  // Then we can create a model instance and bind the tools to it

  const model = new ChatAnthropic({
    model: "claude-3-5-sonnet-20240620",
    temperature: 0,
  }).bindTools(tools);

  // const model = new ChatOpenAI({
  //   modelName: "gpt-4o-mini",
  //   temperature: 0,
  // }).bindTools(tools);



  /**
   * -----------------------------
   * SHOULD CONTINUE NODE
   * the function that determines whether to continue or not
   * -----------------------------
   */
  function shouldContinue(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    // If the LLM makes a tool call, then we route to the "tools" node
    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    // Otherwise, we stop (reply to the user)
    return "__end__";
  }

 
  /**
   * -----------------------------
   * DEFINE MAIN CALLER FUNCTION
   * -----------------------------
   * 1. formats the prompt to be sent to the model
   * 2. injects list of tools, and history of messages from graph state
   * 3. waits for result
   * 4. then return an array of messages with the result, which is what graph state expects and will update for next run
   */
  async function callModel(state: typeof GraphState.State) {

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop. You have access to the following tools: {tool_names}.\n{system_message}\nCurrent time: {time}.`,
      ],
      new MessagesPlaceholder("messages"), // we pass in our message history from state below
    ]);

    const formattedPrompt = await prompt.formatMessages({
      system_message: "You are helpful HR Chatbot Agent.",
      time: new Date().toISOString(),
      tool_names: tools.map((tool) => tool.name).join(", "),
      messages: state.messages, // history of messages from state
    });

    const result = await model.invoke(formattedPrompt);

    return { messages: [result] };
  }



  /**
   * -----------------------------
   * BUILD THE GRAPH
   * -----------------------------
   * 
   * 1. Define the nodes and edges
   * 2. Define the checkpointer for state persistence
   * 3. Compile the graph into a runnable
   * 4. Invoke the runnable with a query
   */

  // Define a new graph - and pass in our function for keeping track of the state
  const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent");

  // Initialize the MongoDB memory to persist state between graph runs
  // const checkpointer = new MongoDBSaver({ client, dbName });

  // This compiles it into a LangChain Runnable.
  // Note that we're passing the memory when compiling the graph


  // export const app = workflow.compile();
  // const app = workflow.compile({ checkpointer });

  let app;

  if (process.env.MODE === "studio") {

    // No checkpointer for studio version since it handles internally 
    app = workflow.compile();

  } else {

    // Initialize the MongoDB memory to persist state between graph runs
    const checkpointer = new MongoDBSaver({ client, dbName });

    app = workflow.compile({ checkpointer });
    
  }

  export { app };
