package xmlProcess;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
//import com.databricks.spark.csv;
import org.sweble.wikitext.engine.EngineException;
import org.sweble.wikitext.engine.PageId;
import org.sweble.wikitext.engine.PageTitle;
import org.sweble.wikitext.engine.WtEngineImpl;
import org.sweble.wikitext.engine.config.WikiConfig;
import org.sweble.wikitext.engine.nodes.EngProcessedPage;
import org.sweble.wikitext.engine.utils.DefaultConfigEnWp;
import org.sweble.wikitext.parser.parser.LinkTargetException;

public class ParseXML {

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("ParseXML").setMaster("local[2]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		
		//Reducing logging for better debugging
		Logger.getRootLogger().setLevel(Level.ERROR);
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);
		
		 DataFrame df = sqlContext.read()
		         .format("xml")
		         .option("rowTag", "page")
		         .load("sample.xml");
		 
		 df.registerTempTable("pages");
		 DataFrame newDF = sqlContext.sql("select title,revision.text from pages where revision.text is not null");
		 
		 //newDF.show();
		 JavaRDD<Row> res = newDF.javaRDD();
		 
		 JavaRDD<String> result = res.map(new Function <Row, String>(){
			 public String call(Row entry) throws LinkTargetException, EngineException {
				 	String title = entry.getString(0);
					String wikiText = entry.getString(1);
					
				    // Set-up a simple wiki configuration
				    WikiConfig config = DefaultConfigEnWp.generate();
				    // Instantiate a compiler for wiki pages
				    WtEngineImpl engine = new WtEngineImpl(config);
				    // Retrieve a page
				    PageTitle pageTitle = PageTitle.make(config, title);
				    PageId pageId = new PageId(pageTitle, -1);
				    // Compile the retrieved page
				    EngProcessedPage cp = engine.postprocess(pageId, wikiText, null);
				    TextConverter p = new TextConverter(config, Integer.MAX_VALUE);
				    
				    String processedText = (String)p.go(cp.getPage());
				    
				    return preprocess(processedText);
			 }	 
		 });
		 
		 result.saveAsTextFile("output");
		 
		 //System.out.println("##############Number : " + newDF.rdd().count());
		 
		// df.select("revision.text").write().format("com.databricks.spark.csv").save("output.csv");
		// df.show();

	}
	
	public static String preprocess(String text){
		//TODO
	/*	1. Stemming
		2. TF-IDF*/
		
		return text;
	}
	
	

}



/*public static String markdown(Row line){
	String entry = line.toString();
	
	
	 StringWriter writer = new StringWriter();

        HtmlDocumentBuilder builder = new HtmlDocumentBuilder(writer);
        builder.setEmitAsDocument(false);

        MarkupParser parser = new MarkupParser(new MediaWikiDialect());
        parser.setBuilder(builder);
        parser.parse(markup);

        final String html = writer.toString();
        final StringBuilder cleaned = new StringBuilder();

        HTMLEditorKit.ParserCallback callback = new HTMLEditorKit.ParserCallback() {
                public void handleText(char[] data, int pos) {
                    cleaned.append(new String(data)).append(' ');
                }
        };
        new ParserDelegator().parse(new StringReader(html), callback, false);
	
	return null;
}
*/