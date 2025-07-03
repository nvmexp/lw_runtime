<!-- XSLT script to colwert doxygen define xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
  <html>
  <body>
    <!-- Load all doxgen generated xml files -->
    <xsl:for-each select="//memberdef[@kind='define']">
      <b>Define <xsl:value-of select="name"/>
      <xsl:if test="param">(</xsl:if>
      <xsl:for-each select="param">
        <xsl:value-of select="defname"/>
        <xsl:if test="position() != last()">, </xsl:if>
      </xsl:for-each>
      <xsl:if test="param">)</xsl:if>
      </b>

        <xsl:for-each select="briefdescription/para">
            <p style="margin-left:80px"><xsl:value-of select="."/></p>
        </xsl:for-each>
        <xsl:for-each select="detaileddescription/para">
            <p style="margin-left:80px"><xsl:value-of select='./text()[normalize-space()][1]'/></p>
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
