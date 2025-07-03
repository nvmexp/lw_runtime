<!-- XSLT script to colwert doxygen define xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="xml" indent="yes"/>
  <xsl:preserve-space elements="initializer"/>
  <xsl:template match="/">
  <html>
  <body>
    <!-- Load all doxgen generated xml files -->
    <xsl:for-each select="//memberdef[@kind='variable']">
      <b><xsl:value-of select="definition"/></b>
      <br/>
      <pre>
        <xsl:value-of select="initializer"/>
      </pre>
      <br/>

        <xsl:for-each select="briefdescription/para">
            <p style="margin-left:80px"><xsl:value-of select="."/></p>
        </xsl:for-each>
        <xsl:for-each select="detaileddescription/para">
            <p style="margin-left:80px"><xsl:value-of select='./node()[not(self::jama-implements)]'/></p>
        </xsl:for-each>

        <br/>

        <h4 style="margin-left:40px;color:#76b900;">File Location</h4>
        <p style="margin-left:80px">
            <xsl:value-of select="location/@file"/>
        </p>
    </xsl:for-each>
  </body>
  </html>
  </xsl:template>
</xsl:stylesheet>
